#!/usr/bin/env python

# @Authors: Lukas Burget, Mireia Diez, Dominik Klement (burget@fit.vutbr.cz, mireia@fit.vutbr.cz, xkleme15@vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Revision History
#   L. Burget   20/1/2021 1:00AM - original version derived from the more
#                                  complex VB_diarization.py avaiable at
# https://github.com/BUTSpeechFIT/VBx/blob/e39af548bb41143a7136d08310765746192e34da/VBx/VB_diarization.py
#

import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from VBx.enums.losses import AvailableLosses
from VBx.enums.backprop_types import AvailableBackpropTypes
from VBx.losses.bce import BCELoss
from VBx.losses.ede import EDELoss


class DiscriminativeVB(nn.Module):
    def __init__(self,
                 device: torch.device,
                 config,
                 phi: torch.double,
                 fa: torch.double,
                 fb: torch.double,
                 loop_p: torch.double,
                 tr: torch.double,
                 init_smoothing: float,
                 epsilon: torch.double = 1e-4,
                 alpha_qinit: torch.double = 1.0,
                 alpha: torch.double = None,
                 invL: torch.double = None,
                 use_gmm: bool = False) -> None:

        super(DiscriminativeVB, self).__init__()

        self.device = device
        self._config = config
        self.use_gmm = use_gmm

        lp = loop_p * torch.ones(1, 1)
        lp_logit = torch.log(lp / (1 - lp))
        self.loop_prob_logit = nn.Parameter(lp_logit * torch.ones((1, 1), dtype=torch.double),
                                            requires_grad='lp' in self._config.trainable_params and \
                                                          not self.use_gmm and loop_p > 0)
        self.fa = nn.Parameter(fa * torch.ones((1, 1), dtype=torch.double),
                               requires_grad='fa' in self._config.trainable_params)
        self.fb = nn.Parameter(fb * torch.ones((1, 1), dtype=torch.double),
                               requires_grad='fb' in self._config.trainable_params)

        self.log_ism = nn.Parameter(torch.log(init_smoothing * torch.ones((1, 1), dtype=torch.double)),
                                    requires_grad='ism' in self._config.trainable_params)

        # Diagonal matrix of eigenvalues
        # self.phi = nn.Parameter(phi, requires_grad='phi' in self._config.trainable_params)
        self.original_log_phi = torch.clone(torch.log(phi)).detach()
        self.log_phi = nn.Parameter(torch.log(phi), requires_grad='phi' in self._config.trainable_params)

        # Transition matrix
        self.original_tr = torch.clone(tr).detach()
        self.tr = nn.Parameter(tr, requires_grad='tr' in self._config.trainable_params)

        train_loss_weights = self._config.use_loss_weights and \
                             self._config.backprop_type == AvailableBackpropTypes.AFTER_EACH_ITER and \
                             'loss_weights' in self._config.trainable_params

        if self._config.use_sigmoid_loss_weights:
            loss_weights_tensor = torch.zeros(
                (self._config.max_iters if self._config.avg_last_n_iters == -1 else self._config.avg_last_n_iters,),
                dtype=torch.double
            )
        else:
            loss_weights_tensor = torch.ones(
                (self._config.max_iters if self._config.avg_last_n_iters == -1 else self._config.avg_last_n_iters,),
                dtype=torch.double
            )
        self.loss_weights = nn.Parameter(loss_weights_tensor, requires_grad=train_loss_weights)

        train_loss_scale = 'loss_scale' in self._config.trainable_params and \
                           self._config.use_loss_scale and not self._config.use_adaptive_scale
        self.log_loss_scale = nn.Parameter(
            torch.log(torch.ones((1, 1), dtype=torch.double) * self._config.initial_loss_scale),
            requires_grad=train_loss_scale)

        self.max_iters = self._config.max_iters
        self.epsilon = epsilon
        self.alpha_qinit = alpha_qinit

        # Buffer that stores log_gamma values throughout the VB iterations
        self.log_gamma_buffer = []

        if alpha is not None:
            self.alpha = torch.tensor(alpha.copy(), dtype=torch.double, requires_grad=False)
        else:
            self.alpha = None

        if invL is not None:
            self.invL = torch.tensor(invL.copy(), dtype=torch.double, requires_grad=False)
        else:
            self.invL = None

        # Normal loss includes scaling and other postprocessing
        self._loss = self._setup_loss()

        # Raw loss is computed using the output gammas without any additional postprocessing
        self._raw_loss = self._setup_raw_loss()
        self._optimizer = self._setup_optimizer()

    def forward(self, X, pi, gamma):  # -> Tuple[torch.Tensor, torch.Tensor, List]
        self.log_gamma_buffer = []
        return self.VBx(X, pi, torch.nn.functional.softmax(torch.exp(self.log_ism) * gamma, dim=1))

    def compute_KL(self):
        Ew = torch.linalg.inv(torch.mm(self.original_tr.T, self.original_tr)).detach()
        Eb = torch.linalg.inv(
            torch.mm(self.original_tr.T / torch.exp(self.original_log_phi), self.original_tr)).detach()
        k = Ew.shape[0]

        tr_t_tr = torch.mm(self.tr.T, self.tr)
        KL_Ew = 0.5 * (-torch.logdet(torch.mm(Ew, tr_t_tr)) + torch.trace(torch.mm(Ew, tr_t_tr)) - k)
        KL_Eb = 0.5 * (torch.sum(self.log_phi) - torch.logdet(torch.mm(Eb, tr_t_tr)) + torch.trace(
            torch.mm(Eb, torch.mm(self.tr.T / torch.exp(self.log_phi), self.tr))) - k)

        return KL_Ew, KL_Eb

    def compute_loss(self, y_labels):
        if not self.log_gamma_buffer:
            raise 'Forward pass needs to be executed before computing a loss.'

        if not self.training:
            current_loss = self._loss(self.log_gamma_buffer[-1], y_labels)
            current_raw_loss = self._raw_loss(self.log_gamma_buffer[-1].detach(), y_labels)
            return current_loss.detach(), current_raw_loss.detach()

        if self._config.backprop_type == AvailableBackpropTypes.AFTER_CONVERGENCE:
            current_loss = self._loss(self.log_gamma_buffer[-1], y_labels)
            current_raw_loss = self._raw_loss(self.log_gamma_buffer[-1].detach(), y_labels)
        elif self._config.backprop_type == AvailableBackpropTypes.AFTER_EACH_ITER:
            losses = torch.stack(
                [self._loss(x, y_labels) for x in self.log_gamma_buffer[-self._config.avg_last_n_iters:]])
            raw_losses = torch.stack(
                [self._raw_loss(x.detach(), y_labels) for x in self.log_gamma_buffer[-self._config.avg_last_n_iters:]])

            if self._config.use_loss_weights:
                if self._config.use_sigmoid_loss_weights:
                    sgm_weights = F.sigmoid(self.loss_weights)
                    sgm_weights_2 = [(1 - torch.prod(sgm_weights[:i], dtype=torch.double)) for i in
                                     range(1, len(self.loss_weights) + 1)]
                    sws = sum(sgm_weights_2)
                    for i in range(len(losses)):
                        losses[i] *= sgm_weights_2[i] / sws
                else:
                    losses *= F.softmax(self.loss_weights, dim=0)

            current_loss = torch.sum(losses)
            current_raw_loss = torch.sum(raw_losses)

            if not self._config.use_loss_weights:
                current_loss /= losses.size()[0]
                current_raw_loss /= raw_losses.size()[0]
        else:
            raise 'Backprop type not implemented.'

        if self._config.use_regularization:
            # KL regularization
            KL_Ew, KL_Eb = self.compute_KL()
            current_loss += self._config.regularization_coeff_ew * KL_Ew + self._config.regularization_coeff_eb * KL_Eb

        return current_loss, current_raw_loss.detach()

    def process_gammas(self, log_gammas):
        """
        Processes the gammas the same way as loss does.
        """
        if self._config.loss == AvailableLosses.EDE:
            return self._loss._process_decisions(log_gammas)

        return torch.sigmoid(self._loss._process_decisions(log_gammas))

    def optimize_params(self, current_loss):
        self._optimizer.zero_grad()
        current_loss.backward()
        self._optimizer.step()

    def tb_log_params(self, tb_writer: SummaryWriter, tag, suffix, step):
        tb_writer.add_scalar(f'{tag}/fa_{suffix}', self.fa, step)
        tb_writer.add_scalar(f'{tag}/fb_{suffix}', self.fb, step)
        tb_writer.add_scalar(f'{tag}/pl_{suffix}', torch.sigmoid(self.loop_prob_logit), step)
        tb_writer.add_scalar(f'{tag}/loss_scale_{suffix}', torch.exp(self.log_loss_scale), step)
        tb_writer.add_histogram(f'{tag}/log_phi_{suffix}', self.log_phi, step)
        tb_writer.add_histogram(f'{tag}/tr_{suffix}', self.tr, step)

        tb_writer.add_scalar(f'{tag}/ism_{suffix}', torch.exp(self.log_ism), step)

        plt.clf()
        if self._config.use_sigmoid_loss_weights:
            sgm_weights = F.sigmoid(self.loss_weights.detach())
            sgm_weights_2 = [torch.prod(sgm_weights[:i]) for i in range(1, len(self.loss_weights) + 1)]
            plt.bar(x=range(len(self.loss_weights)), height=(1 - np.array(sgm_weights_2)))
        else:
            plt.bar(x=range(len(self.loss_weights)),
                    height=torch.nn.functional.softmax(self.loss_weights.detach(), dim=0).numpy())
        tb_writer.add_figure(f'{tag}/loss_weights_{suffix}', plt.gcf(), step)

    def tb_log_params_grads(self, tb_writer, tag, suffix, step):
        if self.fa.requires_grad:
            tb_writer.add_scalar(f'{tag}/fa_{suffix}', self.fa.grad, step)
        if self.fb.requires_grad:
            tb_writer.add_scalar(f'{tag}/fb_{suffix}', self.fb.grad, step)
        if self.loop_prob_logit.requires_grad:
            tb_writer.add_scalar(f'{tag}/pl_logit_{suffix}', self.loop_prob_logit.grad, step)
        if self.log_phi.requires_grad:
            tb_writer.add_scalar(f'{tag}/log_phi_l2_{suffix}', torch.norm(self.log_phi.grad), step)
            tb_writer.add_histogram(f'{tag}/log_phi_{suffix}', self.log_phi.grad, step)
        if self.tr.requires_grad:
            tb_writer.add_scalar(f'{tag}/tr_l2_{suffix}', torch.norm(self.tr.grad), step)
            tb_writer.add_histogram(f'{tag}/tr_{suffix}', self.tr.grad, step)
        if self.log_loss_scale.requires_grad:
            tb_writer.add_scalar(f'{tag}/log_loss_scale_{suffix}', self.log_loss_scale.grad, step)
        if self.loss_weights.requires_grad:
            tb_writer.add_scalar(f'{tag}/loss_weights_l2_{suffix}', torch.norm(self.loss_weights.grad), step)

    def save_model(self, path, ep, step):
        return torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'epoch': ep,
            'step': step
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        if 'ism' in checkpoint['model_state_dict']:
            checkpoint['model_state_dict']['log_ism'] = torch.log(checkpoint['model_state_dict']['ism'])
            del checkpoint['model_state_dict']['ism']
        self.load_state_dict(checkpoint['model_state_dict'])
        # self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['step']

    def VBx(self, X_in, pi_in, gamma):
        """
        Inputs:
        X           - T x D array, where columns are D dimensional feature vectors
                      (e.g. x-vectors) for T frames
        Phi         - D array with across-class covariance matrix diagonal.
                      The model assumes zero mean, diagonal across-class and
                      identity within-class covariance matrix.
        loopProb    - Probability of not switching speakers between frames
        Fa          - Scale sufficient statiscits
        Fb          - Speaker regularization coefficient Fb controls the final number of speakers
        pi          - If integer value, it sets the maximum number of speakers
                      that can be found in the utterance.
                      If vector, it is the initialization for speaker priors (see Outputs: pi)
        gamma       - An initialization for the matrix of responsibilities (see Outputs: gamma)
        maxIters    - The maximum number of VB iterations
        epsilon     - Stop iterating, if the obj. fun. improvement is less than epsilon
        alphaQInit  - Dirichlet concentraion parameter for initializing gamma
        ref         - T dim. integer vector with per frame reference speaker IDs (0:maxSpeakers)
        plot        - If set to True, plot per-frame marginal speaker posteriors 'gamma'
        return_model- Return also speaker model parameter
        alpha, invL - If provided, these are speaker model parameters used in the first iteration

        Outputs:
        gamma       - S x T matrix of responsibilities (marginal posteriors)
                      attributing each frame to one of S possible speakers
                      (S is defined by input parameter pi)
        pi          - S dimensional column vector of ML learned speaker priors.
                      This allows us to estimate the number of speaker in the
                      utterance as the probabilities of the redundant speaker
                      converge to zero.
        Li          - Values of auxiliary function (and DER and frame cross-entropy
                      between gamma and reference, if 'ref' is provided) over iterations.
        alpha, invL - Speaker model parameters returned only if return_model=True

        Reference:
          Landini F., Profant J., Diez M., Burget L.: Bayesian HMM clustering of
          x-vector sequences (VBx) in speaker diarization: theory, implementation
          and analysis on standard tasks
        """
        """
        The comments in the code refers to the equations from the paper above. Also
        the names of variables try to be consistent with the symbols in the paper.
        """
        X = torch.mm(X_in, self.tr.T)
        D = X.shape[1]

        if type(pi_in) is int:
            pi = torch.ones(pi_in, dtype=torch.double, requires_grad=False) / pi_in
        else:
            pi = pi_in

        if gamma is None:
            # initialize gamma from flat Dirichlet prior with
            # concentration parameter alphaQInit
            gamma = torch.from_numpy(np.random.gamma(self.alpha_qinit, size=(X.shape[0], len(pi))))
            gamma = gamma / gamma.sum(1, keepdim=True)

        assert (gamma.size()[1] == len(pi) and gamma.size()[0] == X.shape[0])

        phi = torch.exp(self.log_phi)

        G = -0.5 * (torch.sum(X ** 2, dim=1, keepdim=True) + D * torch.log(
            2 * torch.tensor(np.pi)))  # per-frame constant term in (23)
        V = torch.sqrt(phi)  # between (5) and (6)
        rho = X * V  # (18)
        Li = []
        log_gamma = torch.log(gamma)

        # We don't need transition matrix for GMM.
        if not self.use_gmm:
            loop_prob = torch.sigmoid(self.loop_prob_logit)
            if self._config.use_full_tr:
                tr = torch.eye(len(pi)) * loop_prob
                tr_off_diag = ((torch.ones_like(tr) - torch.eye(len(pi))) * ((1 - loop_prob) / (len(pi) - 1)))
                tr += tr_off_diag
            else:
                tr = torch.eye(len(pi)) * loop_prob + (1 - loop_prob) * pi  # (1) transition probability matrix

        for ii in range(self.max_iters if self.training else self._config.eval_max_iters):
            self.invL = 1.0 / (1 + self.fa / self.fb * gamma.sum(dim=0, keepdim=True).T * phi)  # (17) for all speakers
            self.alpha = self.fa / self.fb * self.invL * torch.tensordot(gamma.T, rho, dims=1)  # (16) for all speakers
            log_p_ = self.fa * (torch.tensordot(rho, self.alpha.T, dims=1) - 0.5 * torch.tensordot(
                (self.invL + self.alpha ** 2), phi, dims=1) + G)  # (23) for all speakers

            if self.use_gmm:
                log_gamma, gamma, log_pX_, pi = self._compute_gmm(pi, log_p_)
            else:
                log_gamma, gamma, log_pX_, pi, tr = self._compute_hmm(pi, log_p_, tr)

            ELBO = (log_pX_ + self.fb * 0.5 * torch.sum(
                torch.log(self.invL) - self.invL - self.alpha ** 2 + 1)).detach()  # (25)
            Li.append([ELBO])
            self.log_gamma_buffer.append(log_gamma)

            if ii > 0 and ELBO - Li[-2][0] < self.epsilon:
                if ELBO - Li[-2][0] < 0:
                    print('WARNING: Value of auxiliary function has decreased!')
                if (self.training and self._config.early_stop_vb) or (
                        not self.training and self._config.eval_early_stop_vb):
                    break
        return log_gamma, gamma, pi, Li

    def _compute_hmm(self, pi, log_p_, tr):
        loop_prob = torch.sigmoid(self.loop_prob_logit)
        if not self._config.use_full_tr:
            tr = torch.eye(len(pi)) * loop_prob + (1 - loop_prob) * pi  # (1) transition probability matrix

        # (19) gamma, (20) logA, (21) logB, (22) log_pX_
        log_gamma, log_pX_, logA, logB = DiscriminativeVB._forward_backward(log_p_, tr, pi)
        gamma = torch.exp(log_gamma)

        if self._config.use_full_tr:
            pi = gamma[0]
            tr = tr * torch.sum(torch.exp(torch.unsqueeze(logA[:-1, :], 2) +
                                          torch.unsqueeze(logB[1:, :] + log_p_[1:, :], 1) - log_pX_), dim=0)
            tr /= torch.unsqueeze(torch.sum(tr, dim=1), dim=1)
        else:
            pi = (gamma[0] + (1 - loop_prob) * pi * torch.sum(torch.exp(
                torch.logsumexp(logA[:-1], dim=1, keepdim=True) + log_p_[1:] + logB[1:] - log_pX_),
                dim=0)).reshape(-1)  # (24)
            pi = pi / pi.sum()

        return log_gamma, gamma, log_pX_, pi, tr

    def _compute_gmm(self, pi, log_p_):
        eps = 1e-8
        lpi = torch.log(pi)

        log_p_x = torch.logsumexp(log_p_ + lpi, dim=-1)  # expected (through q(Y)) per-frame log likelihood
        log_pX_ = torch.sum(log_p_x, dim=0)  # expected (through q(Y)) recording log likelihood
        log_gamma = log_p_ + lpi - torch.unsqueeze(log_p_x, dim=1)
        gamma = torch.exp(log_gamma)

        if self._config.use_full_tr:
            raise Exception('Not Implemented.')
        else:
            # PI cannot be equal directly to 0, because log_pi would be -inf, and it would result in nan values.
            pi = torch.sum(gamma, dim=0) + eps
            pi /= torch.sum(pi, dim=0)

        return log_gamma, gamma, log_pX_, pi

    @staticmethod
    def _forward_backward(lls, tr, ip):
        """
        Inputs:
            lls - matrix of per-frame log HMM state output probabilities
            tr  - transition probability matrix
            ip  - vector of initial state probabilities (i.e. starting in the state)
        Outputs:
            pi  - matrix of per-frame state occupation posteriors
            tll - total (forward) log-likelihood
            lfw - log forward probabilities
            lfw - log backward probabilities
        """
        eps = 1e-8
        ltr = torch.log(tr)
        lfw = [torch.full_like(lls[0], -torch.inf) for _ in range(lls.size()[0])]
        lbw = [torch.full_like(lls[0], -torch.inf) for _ in range(lls.size()[0])]
        lfw[0] = lls[0] + torch.log(ip + eps)
        lbw[-1] = torch.zeros_like(lls[0])

        for ii in range(1, len(lls)):
            lfw[ii] = lls[ii] + torch.logsumexp(lfw[ii - 1] + ltr.T, dim=1)

        for ii in reversed(range(len(lls) - 1)):
            lbw[ii] = torch.logsumexp(ltr + lls[ii + 1] + lbw[ii + 1], dim=1)

        lfw = torch.stack(lfw, dim=0)
        lbw = torch.stack(lbw, dim=0)

        tll = torch.logsumexp(lfw[-1], dim=0)
        log_gamma = lfw + lbw - tll
        return log_gamma, tll, lfw, lbw

    def _setup_loss(self):
        if self._config.loss == AvailableLosses.BCE:
            return BCELoss(use_scale=self._config.use_loss_scale, log_loss_scale=self.log_loss_scale, use_logprobs=True)
        elif self._config.loss == AvailableLosses.EDE:
            return EDELoss(use_scale=self._config.use_loss_scale, log_loss_scale=self.log_loss_scale, use_logprobs=True)
        else:
            raise 'Loss not implemented'

    def _setup_raw_loss(self):
        if self._config.loss == AvailableLosses.BCE:
            return BCELoss(use_scale=False, log_loss_scale=1, use_logprobs=True)
        elif self._config.loss == AvailableLosses.EDE:
            return EDELoss(use_scale=False, log_loss_scale=1, use_logprobs=True)
        else:
            raise 'Loss not implemented'

    def _setup_optimizer(self):
        model_params = dict(self.named_parameters())
        settings_list = []
        lr_params_settings = self._config.lr.keys()
        all_lr = 1e-2
        if 'all' in self._config.lr:
            settings_list = self.parameters()
            all_lr = self._config.lr['all']
        else:
            set_parameters = set()
            for lr_setting in lr_params_settings:
                # Each key can contain multiple parameter names separated by commas.
                setting_param_names = lr_setting.replace(' ', '').split(',')
                setting_params = []
                for pname in setting_param_names:
                    if pname == 'all' or pname == 'other':
                        continue

                    # config.json contains lp as loop probability, but we internally represent it
                    # as a logit and then run sigmoid to obtain a valid probability.
                    if pname == 'lp':
                        pname = 'loop_prob_logit'

                    if pname == 'phi':
                        pname = 'log_phi'

                    # Save parameters that are already set to know what parameters get "other" lr.
                    set_parameters.add(pname)
                    # Parameters corresponding to the specified names.
                    setting_params.append(model_params[pname])

                # Torch format of param-specific lr settings.
                if setting_params:
                    settings_list.append({
                        'params': setting_params,  # List of parameters.
                        'lr': self._config.lr[lr_setting]  # Corresponding LR.
                    })

            if 'other' in self._config.lr:
                # Save named params to set and do difference to set left params.
                all_params = set(map(lambda x: x[0], self.named_parameters()))
                notset_params = all_params.difference(set_parameters)
                settings_list.append({
                    'params': [model_params[x] for x in notset_params],
                    'lr': self._config.lr['other']
                })

        optimizer = optim.Adam(settings_list, lr=all_lr)
        return optimizer
