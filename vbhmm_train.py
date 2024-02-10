#!/usr/bin/env python

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

import os
from math import ceil
from shutil import copy as shell_copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from socket import gethostname
from datetime import datetime
import yamlargparse

from VBx.dataset.data_loader import create_data_loader
from VBx.utils.kaldi_utils import read_plda
from VBx.models import VBx as VBx
from VBx.dataset.dataset import BasicDataset
from VBx.dataset.data_file_loader import DataFileLoader
from VBx.utils.config import process_args
from VBx.utils.metrics import DER

mpl.rcParams['figure.dpi'] = 150


def read_kaldi_plda(path):
    plda_mu, plda_tr, plda_psi = read_plda(path)
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]
    return plda_mu, plda_psi, plda_tr


def trace_handler(prof):
    print(prof.key_averages().table(row_limit=-1))
    prof.export_chrome_trace("test_trace_" + str(prof.step_num) + ".json")


def plot_gammas(gammas, ys_alig, zero_ones_labels=False):
    colors = ['#61a4f1', '#a7f3e5', '#083167', '#c3a443', '#880b08', '#9f2903', '#7c3d2a',
              '#63d0fc',
              '#edce04', '#fa3dcf', '#82cefb', '#5f54b5', '#65388d', '#03a6d0', '#84cb7a',
              '#485a18',
              '#474c85', '#67edbb', '#85d7e6', '#7fb5ea', '#96591d', '#298b53', '#522f12',
              '#99c542',
              '#3b17f0', '#e4ef46', '#7d4247', '#9ded6f', '#260995', '#64ec6b', '#33f63d',
              '#7e5a90',
              '#448cc8', '#36b23f', '#eafb65', '#99f0e0', '#c059f3', '#02b8b5', '#c3e236',
              '#5f8ed3',
              "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
              "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
              "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
              "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
              "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
              "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
              "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
              "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

              "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
              "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
              "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
              "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
              "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
              "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
              "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
              "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
              ]

    _, axs = plt.subplots(len(gammas), 1)
    ax_ind = -1
    for gamma, y_alig in zip(gammas, ys_alig):
        ax_ind += 1
        label_counter = -1

        for i in range(gamma.shape[1]):
            current_color = colors[i]
            axs[ax_ind].plot(gamma[:, i], color=current_color, linewidth=0.5)

            if y_alig is not None:
                if zero_ones_labels:
                    # If we use zero_ones labels, dashed lines may overlap which makes discriminating
                    # overlapped ground truth labels impossible.
                    if torch.sum(y_alig[:, i] == 1.0) > 0:
                        label_counter += 1

                    axs[ax_ind].plot(torch.clamp_min_(y_alig[:, i] - 0.04 * label_counter, 0.0), '--',
                                     color=current_color, linewidth=0.5)
                else:
                    axs[ax_ind].plot(y_alig[:, i], '--', color=current_color, linewidth=0.5)
                for j in range(gamma.shape[0]):
                    # This line makes the overlapped colors in case of ZERO_ONES, and white in case of
                    # prob. labels.
                    if y_alig[j][i] == 1:
                        axs[ax_ind].axvspan(j - 0.5, j + 0.5, facecolor=current_color, alpha=0.2)


def plot_files(dset, indices, processed=False, zero_ones_labels=False, plot_qinit_only=False):
    """
    If processed is true, it plots the exactly same gammas that are on input to the loss fn.
    """
    with torch.no_grad():
        plt.clf()
        gammas = []
        ys_alig = []
        for sp in indices:
            _, _, fn, x, qinit, y = dset[sp]

            if plot_qinit_only:
                gamma = qinit
            else:
                pi_init = torch.ones(qinit.size()[1], dtype=torch.double, requires_grad=False) / qinit.size()[1]

                if args.run_dist:
                    log_gamma, gamma, pi, Li = ddp_model(x, pi=pi_init, gamma=qinit)
                    if processed:
                        gamma = ddp_model.module.process_gammas(log_gamma)
                else:
                    log_gamma, gamma, pi, Li = model(x, pi=pi_init, gamma=qinit)
                    if processed:
                        gamma = model.process_gammas(log_gamma)

            decisions_t = torch.transpose(gamma, 0, 1)
            cost_mx2 = torch.matmul(1 - decisions_t, y) + torch.matmul(decisions_t, 1 - y)
            pred_alig, ref_alig = linear_sum_assignment(cost_mx2.detach())
            y_alig = y[:, ref_alig]

            gammas.append(gamma)
            ys_alig.append(y_alig)

        plot_gammas(gammas, ys_alig, zero_ones_labels)


def eval_model(model, dset, run_dist, der_wrapper: DER, rank, world_size):
    model.eval()
    with torch.no_grad():
        all_ders = []

        for batch in dset:
            batch_der = []
            for sd, sn, fn, x, qinit, y in batch:
                pi_init = torch.ones(qinit.size()[1],
                                     dtype=torch.double,
                                     requires_grad=False) / qinit.size()[1]

                log_gamma, gamma, pi, Li = model(x, pi=pi_init, gamma=qinit)

                assert (np.all(sd[fn][0] == np.array(sn)))
                start, end = sd[fn][1].T

                labels1st = torch.argmax(gamma, dim=1).numpy()
                der, speech_amount = der_wrapper.der_tensor(start, end, labels1st, fn)
                batch_der.append((fn, der, speech_amount))

            if run_dist:
                batch_ders = [[] for _ in range(world_size)]
                dist.all_gather_object(batch_ders, batch_der)
            else:
                batch_ders = [batch_der]

            if rank == 0:
                all_ders.append(batch_ders)

        if rank == 0:
            # Get rid of repetitive files (sampling w repetitions is used for dist. batching)
            files_dict = dict()
            for batch_ders in all_ders:
                for batch_der in batch_ders:
                    for fn, der, speech_amount in batch_der:
                        files_dict[fn] = (der * speech_amount, speech_amount)

            total_der = sum([files_dict[x][0] for x in files_dict])
            total_speech = sum([files_dict[x][1] for x in files_dict])
            return 100.0 * total_der / total_speech


def get_exp_name(args):
    suffix = datetime.strftime(datetime.now(), '%b%d_%H-%M-%S_') + gethostname()

    if 'dihard' in args.in_trainlist.lower():
        if 'domains' in args.in_trainlist.lower():
            train_dset = 'dihard_' + '.'.join(args.in_trainlist.lower().split('/')[1].split('.')[:-2])
        else:
            train_dset = 'dihard'
    elif 'ch' in args.in_trainlist.lower() or '/train.txt' in args.in_trainlist.lower():
        train_dset = 'callhome'
    elif 'ami' in args.in_trainlist.lower():
        train_dset = 'ami'
    else:
        train_dset = os.path.basename(args.in_trainlist)

    config_name = os.path.basename(args.config_path[0].abs_path).split('.yml')[0]
    back_prop_type_initials = ''.join(map(lambda x: x[0], args.backprop_type.name.lower().split('_')))

    name = f"{config_name}" \
           f"__{train_dset}__" \
           f"{args.gt_label_type}" \
           f"_fa_{args.Fa}" \
           f"_fb_{args.Fb}" \
           f"_pl_{args.loopP}" \
           f"_loss_{args.loss.name.lower()}" \
           f"_ils_{args.initial_loss_scale}" \
           f"_mit_{args.max_iters}" \
           f"_bpt_{back_prop_type_initials}" \
           f"_brc_{args.regularization_coeff_eb}" \
           f"_wrc_{args.regularization_coeff_ew}" \
           f"_ls_{int(args.use_loss_scale)}" \
           f"_reg_{int(args.use_regularization)}" \
           f"_ism_{float(args.init_smoothing)}"
    if args.use_gmm:
        name += '_gmm'
    else:
        name += '_hmm'

    if args.exp_name_tag is not None:
        name += '_tag_' + args.exp_name_tag

    return f'{name}_{suffix}'


if __name__ == '__main__':
    parser = yamlargparse.ArgumentParser()
    parser.add_argument('--in-GTlabels-dir', required=True, type=str,
                        help='Directory with the ground truth xvector labels')
    parser.add_argument('--in-INITlabels-dir', required=True, type=str,
                        help='Directory with the AHC initialization xvector labels')
    parser.add_argument('--in-val-GTlabels-dir', required=False, type=str,
                        help='Directory with the ground truth xvector labels for validation set')
    parser.add_argument('--in-val-INITlabels-dir', required=False, type=str,
                        help='Directory with the AHC initialization xvector labels for validation set')
    parser.add_argument('--in-trainlist', required=True, type=str,
                        help='txt list of training files')
    parser.add_argument('--in-vallist', required=True, type=str,
                        help='txt list of validation files')
    parser.add_argument('--in-train-gt-rttm-dir', required=True, type=str,
                        help='Path to the directory containing training ground truth RTTM files.')
    parser.add_argument('--in-val-gt-rttm-dir', required=True, type=str,
                        help='Path to the directory containing validation ground truth RTTM files.')
    parser.add_argument('--xvec-ark-dir', required=True, type=str,
                        help='Kaldi ark with x-vectors from one or more '
                             'input recordings. Attention: all x-vectors from '
                             'one recording must be in one ark file')
    parser.add_argument('--segments-dir', required=True, type=str,
                        help='File with x-vector timing info. See '
                             'diarization_lib.read_xvector_timing_dict')
    parser.add_argument('--val-xvec-ark-dir', required=False, type=str,
                        help='Kaldi ark with x-vectors from one or more '
                             'input recordings. Attention: all x-vectors from '
                             'one recording must be in one ark file')
    parser.add_argument('--val-segments-dir', required=False, type=str,
                        help='File with x-vector timing info. See '
                             'diarization_lib.read_xvector_timing_dict')
    # Eval Dataset Params
    parser.add_argument('--in-eval-list', required=False, type=str,
                        help='txt list of evaluation files')
    parser.add_argument('--in-eval-GTlabels-dir', required=False, type=str,
                        help='Directory with the ground truth xvector labels')
    parser.add_argument('--in-eval-INITlabels-dir', required=False, type=str,
                        help='Directory with the AHC initialization xvector labels for evaluation set.')
    parser.add_argument('--eval-xvec-ark-dir', required=False, type=str,
                        help='Kaldi ark with x-vectors from one or more '
                             'input recordings. Attention: all x-vectors from '
                             'one recording must be in one ark file')
    parser.add_argument('--eval-segments-dir', required=False, type=str,
                        help='File with x-vector timing info. See '
                             'diarization_lib.read_xvector_timing_dict')
    parser.add_argument('--in-eval-gt-rttm-dir', required=False, type=str,
                        help='Path to the directory containing evaluation ground truth RTTM files.')

    parser.add_argument('--xvec-transform', required=True, type=str,
                        help='path to x-vector transformation h5 file')
    parser.add_argument('--plda-file', required=True, type=str,
                        help='File with PLDA model in Kaldi format used for '
                             'AHC and VB-HMM x-vector clustering')
    parser.add_argument('--threshold', required=True, type=float,
                        help='args.threshold (bias) used for AHC')
    parser.add_argument('--lda-dim', required=True, type=int,
                        help='For VB-HMM, x-vectors are reduced to this '
                             'dimensionality using LDA')
    parser.add_argument('--Fa', required=True, type=float,
                        help='Parameter of VB-HMM (see VBx.VBx)')
    parser.add_argument('--Fb', required=True, type=float,
                        help='Parameter of VB-HMM (see VBx.VBx)')
    parser.add_argument('--loopP', required=True, type=float,
                        help='Parameter of VB-HMM (see VBx.VBx)')
    parser.add_argument('--target-energy', required=False, type=float,
                        default=1.0,
                        help='Parameter affecting AHC if the similarity '
                             'matrix is obtained with PLDA. See '
                             'diarization_lib.kaldi_ivector_plda_scoring_dense')
    parser.add_argument('--init-smoothing', required=False, type=float,
                        default=3,
                        help='AHC produces hard assignments of x-vetors to '
                             'speakers. These are "smoothed" to soft '
                             'assignments as the args.initialization for '
                             'VB-HMM. This parameter controls the amount of '
                             'smoothing. Not so important, high value '
                             '(e.g. 10) is OK  => keeping hard assigment')
    parser.add_argument('--gt-label-type', required=False, type=str,
                        default="SINGLE_SPK",
                        help='Ground truth labels are used as probabilities '
                             'that denote the percentage of presence of each '
                             'speaker\'s speech in the x-vector (PROBABILITIES), '
                             'as 0 or 1 values that denote the presence or absence '
                             'of speech of a speaker in a x-vector (ZERO_ONES) '
                             'or as a indicator of the speaker that speaks the most '
                             'in the x-vector (SINGLE_SPK)')
    parser.add_argument('-c', '--config-path', required=True, type=str, action=yamlargparse.ActionConfigFile,
                        help='Path to config.json file.')
    parser.add_argument('--run-dist', required=False, default=False, action='store_true',
                        help='If present, the training has to be run with torchrun.')
    parser.add_argument('--continue-log-dir', required=False,
                        help='Tensorboard log dir to continue the training process from.')
    parser.add_argument('--eval-after-steps', required=False, type=int, default=-1,
                        help='Number of update steps (batches) to evaluate the model on validation data after.')
    parser.add_argument('--eval-after-epochs', required=False, type=int, default=20,
                        help='Number of epochs after which the model is going to be evaluated on train, dev, eval'
                             'datasets. This argument can be overridden by --eval-after-steps, which means that '
                             'the model will be evaluated after --eval-after-epochs number of epochs only if '
                             '--eval-after-steps is not present (i.e. set to -1).')
    parser.add_argument('--save-checkpoint-after-steps', required=False, type=int, default=-1,
                        help='Saves model checkpoint after n update steps (batches).')
    parser.add_argument('--tb-path', required=False, type=str,
                        help='Path to the tensorboard logging dir. If not provided, a default one will be used, '
                             'which consists of date and hostname.')
    parser.add_argument('--num-threads-per-worker', type=int, default=1,
                        help='Number of torch threads used for tensor operations.')
    parser.add_argument('--use-gmm', required=False, default=False, action='store_true',
                        help='If present, GMM is used instead of HMM, i.e. ploop=0.')
    parser.add_argument('--plot-gammas', required=False, default=False, action='store_true',
                        help='Gamma plots of pre-selected files are logged into TensorBoard if the option is present.')
    parser.add_argument('--exp-name-tag', required=False, type=str,
                        help='It helps with classifying the experiments into groups/tags to better navigate through '
                             'TB search.')
    parser.add_argument('--init-model-path', required=False, type=str,
                        help='Path to initial model. If loaded, Fa, Fb, loopP will be overridden.')

    # YAML config parameters
    parser.add_argument('--avg-last-n-iters', type=int, default=-1,
                        help='Dictates how many last VB iterations are averaged to compute gradients '
                             '(in case of using after_each_iter backprop_type)')
    parser.add_argument('--backprop-type', type=str, default='after_each_iter',
                        help='Gradient computation type: after_convergence, or after_each_iter.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size.')
    parser.add_argument('--early-stop-vb', type=bool, default=False,
                        help='If True, VB inference is stopped during training if ELBO stops increasing.')
    parser.add_argument('--eval-max-iters', type=int, default=40,
                        help='Max number of VB iterations used during inference/evaluation.')
    parser.add_argument('--eval-early-stop-vb', type=bool, default=True,
                        help='Same as --early-stop-vb but during evaluation.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of passes through the whole training set.')
    parser.add_argument('--initial-loss-scale', type=float, default=1,
                        help='Initial value of loss scale (calibration).')
    parser.add_argument('--loss', type=str, default='EDE', help='Training loss, either BCE or DER.')
    parser.add_argument('--lr', help='Learning rates.')
    parser.add_argument('--max-iters', type=int, default=10,
                        help='Max number of VB iterations during training (if early_stop_vb is False, than it will '
                             'run --max-iters of VB iterations no matter the ELBO convergence')
    parser.add_argument('--regularization-coeff-eb', default=0, type=float,
                        help='KL divergence between-speaker covariance matrix regularization coefficient.')
    parser.add_argument('--regularization-coeff-ew', default=0, type=float,
                        help='KL divergence within-speaker covariance matrix regularization coefficient.')
    parser.add_argument('--trainable-params', default=['fa', 'fb', 'lp', 'ism'], nargs='*',
                        help='List of trainable parameters.')
    parser.add_argument('--use-full-tr', type=bool, default=False,
                        help='Use full transition matrix (i.e. allow HMM to have connection '
                             'between all pairs of nodes).')
    parser.add_argument('--use-loss-scale', type=bool, default=False,
                        help='If true, loss scale (calibration) is used.')
    parser.add_argument('--use-loss-weights', type=bool, default=False,
                        help='Gradients are computed as a weighted average of the VB iteration gradients.')
    parser.add_argument('--use-regularization', type=bool, default=False,
                        help='If True, KL divergence regularization towards the original (generatively trained) '
                             'PLDA covariance matrices is used.')
    parser.add_argument('--use-sigmoid-loss-weights', type=bool, default=False,
                        help='If true, loss weights for iteration i are computed as 1 - product of '
                             'weights for weights i,i+1, ..., n.')

    args = parser.parse_args()

    if args.run_dist:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Start running basic DDP example on rank {rank}.")
    else:
        rank = 0
        world_size = 1

    # Continue training, i.e. use the previously set config file.
    if args.continue_log_dir is not None:
        config_fname = list(filter(lambda x: '.yml' in x, os.listdir(args.continue_log_dir)))[0]
        args = parser.parse_args(['-c', f'{args.continue_log_dir}/{config_fname}'])

    process_args(args)

    assert 0 <= args.loopP <= 1, f'Expecting loopP between 0 and 1, got {args.loopP} instead.'
    plda_mu, plda_psi, plda_tr = read_kaldi_plda(args.plda_file)

    torch.set_num_threads(args.num_threads_per_worker)

    if rank == 0:
        if args.continue_log_dir is None:
            tb_path = args.tb_path
            if not tb_path:
                tb_path = os.getcwd() + '/runs/' + get_exp_name(args)
            tb_writer = SummaryWriter(log_dir=tb_path)
        else:
            tb_writer = SummaryWriter(log_dir=args.continue_log_dir)

        if args.continue_log_dir is None:
            tb_writer.add_hparams({
                'fa': args.Fa,
                'fb': args.Fb,
                'ploop': args.loopP,
                'max_iters': args.max_iters,
                'loss': args.loss.name,
                'use_full_tr': args.use_full_tr,
                'use_loss_scale': args.use_loss_scale,
                'use_loss_weights': args.use_loss_weights,
                'batch_size': args.batch_size,
                'backprop_type': args.backprop_type.name,
                'init_smoothing': args.init_smoothing,
                'gt_label_type': args.gt_label_type,
                'in_trainlist': args.in_trainlist,
                'use_gmm': args.use_gmm
            }, {
                'der': 0
            })
            shell_copy(args.config_path[0].abs_path, tb_writer.log_dir)

    # We want equal batches across workers.
    assert args.batch_size % world_size == 0, 'Batch size have to be divisible by the number of processes.'

    model = VBx.DiscriminativeVB(device=torch.device("cpu"),
                                 phi=torch.from_numpy(plda_psi[:args.lda_dim].copy()),
                                 tr=torch.from_numpy(plda_tr.copy()),
                                 fa=args.Fa,
                                 fb=args.Fb,
                                 loop_p=args.loopP,
                                 config=args,
                                 init_smoothing=args.init_smoothing,
                                 epsilon=1e-6,
                                 use_gmm=args.use_gmm)

    if args.init_model_path is not None:
        model.load_model(args.init_model_path)

    step_counter = 0
    starting_epoch = 1
    if args.continue_log_dir is not None:
        max_ep = max(*[int(x.split('ep_')[1][:-3]) for x in os.listdir(f'{args.continue_log_dir}/checkpoints')])
        starting_epoch, step_counter = model.load_model(f'{args.continue_log_dir}/checkpoints/model_ep_{max_ep}.pt')
        starting_epoch += 1
        step_counter += 1

    if args.run_dist:
        ddp_model = DDP(model, find_unused_parameters=False)

    train_list = np.loadtxt(args.in_trainlist, dtype=object)
    val_list = np.loadtxt(args.in_vallist, dtype=object)

    der_train = DER(train_list, args.in_train_gt_rttm_dir)
    der_val = DER(val_list, args.in_val_gt_rttm_dir)

    if args.in_eval_gt_rttm_dir:
        eval_list = np.loadtxt(args.in_eval_list, dtype=object)
        der_eval = DER(eval_list, args.in_eval_gt_rttm_dir)
    else:
        eval_list = []
        der_eval = None

    dfl = DataFileLoader(plda_mu,
                         args.segments_dir,
                         args.in_INITlabels_dir,
                         args.xvec_ark_dir,
                         args.in_GTlabels_dir,
                         args.xvec_transform,
                         args.init_smoothing,
                         args.gt_label_type)

    # In case some validation file is not provided, use the val part of the training set.
    if args.in_val_INITlabels_dir is None or \
            args.in_val_GTlabels_dir is None or \
            args.val_xvec_ark_dir is None or \
            args.val_segments_dir is None:
        val_dfl = dfl
    else:
        val_dfl = DataFileLoader(plda_mu,
                                 args.val_segments_dir,
                                 args.in_val_INITlabels_dir,
                                 args.val_xvec_ark_dir,
                                 args.in_val_GTlabels_dir,
                                 args.xvec_transform,
                                 args.init_smoothing,
                                 args.gt_label_type)

    # Eval rttm dir arg presence determines whether to score the Eval dataset or not.
    if args.in_eval_gt_rttm_dir:
        eval_dfl = DataFileLoader(plda_mu,
                                  args.eval_segments_dir,
                                  args.in_eval_INITlabels_dir,
                                  args.eval_xvec_ark_dir,
                                  args.in_eval_GTlabels_dir,
                                  args.xvec_transform,
                                  args.init_smoothing,
                                  args.gt_label_type,
                                  eval_mode=True)
        eval_dset = BasicDataset(eval_list, eval_dfl)
    else:
        eval_dset = None

    train_dset = BasicDataset(train_list, dfl)
    val_dset = BasicDataset(val_list, val_dfl)

    val_steps = args.eval_after_steps
    if val_steps == -1:
        val_steps = ceil(len(train_list) / args.batch_size) * args.eval_after_epochs

    save_steps = args.save_checkpoint_after_steps
    if save_steps == -1:
        save_steps = ceil(len(train_list) / args.batch_size)

    if args.run_dist:
        train_dsampler = DistributedSampler(train_dset,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=True)
        train_dl = DataLoader(train_dset,
                              batch_size=args.batch_size // world_size,
                              collate_fn=lambda x: x,
                              sampler=train_dsampler)
        val_dl = DataLoader(val_dset,
                            batch_size=args.batch_size // world_size,
                            shuffle=False,
                            collate_fn=lambda x: x,
                            sampler=DistributedSampler(val_dset,
                                                       num_replicas=world_size,
                                                       rank=rank,
                                                       shuffle=False))

        if args.in_eval_gt_rttm_dir:
            eval_dl = DataLoader(eval_dset,
                                 batch_size=args.batch_size // world_size,
                                 shuffle=False,
                                 collate_fn=lambda x: x,
                                 sampler=DistributedSampler(eval_dset,
                                                            num_replicas=world_size,
                                                            rank=rank,
                                                            shuffle=False))
    else:
        train_dl = create_data_loader(train_dset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        val_dl = create_data_loader(val_dset,
                                    batch_size=args.batch_size,
                                    shuffle=False)

        if args.in_eval_gt_rttm_dir:
            eval_dl = create_data_loader(eval_dset,
                                         batch_size=args.batch_size,
                                         shuffle=False)

    if args.run_dist:
        train_der = eval_model(ddp_model, train_dl, args.run_dist, der_train, rank, world_size)
        val_der = eval_model(ddp_model, val_dl, args.run_dist, der_val, rank, world_size)

        if args.in_eval_gt_rttm_dir:
            eval_der = eval_model(ddp_model, eval_dl, args.run_dist, der_eval, rank, world_size)
    else:
        train_der = eval_model(model, train_dl, args.run_dist, der_train, rank, world_size)
        val_der = eval_model(model, val_dl, args.run_dist, der_val, rank, world_size)

        if args.in_eval_gt_rttm_dir:
            eval_der = eval_model(model, eval_dl, args.run_dist, der_eval, rank, world_size)

    # Plot init gammas
    if rank == 0:
        if not args.continue_log_dir:
            tb_writer.add_scalar('der/train_ep', train_der, 0)
            tb_writer.add_scalar('der/val_ep', val_der, 0)

            if args.in_eval_gt_rttm_dir:
                tb_writer.add_scalar('der/eval_ep', eval_der, 0)

        if 'dihard_exp' in args.in_GTlabels_dir:
            train_file_indices = [78, 35, 90]
            val_file_indices = [2, 5, 1, 3, 4]
        elif 'ch' in args.in_GTlabels_dir:
            train_file_indices = [1, 5, 185]
            val_file_indices = [2, 5, 1, 3, 4]
        elif 'ami' in args.in_GTlabels_dir:
            train_file_indices = [1, 2, 3]
            val_file_indices = [2, 5, 1, 3, 4]
        else:
            train_file_indices = [0, 1, 2, 3]
            val_file_indices = [0, 1, 2, 3]

        zero_ones_labels = args.gt_label_type == 'ZERO_ONES'

        if args.plot_gammas:
            plot_files(train_dset, train_file_indices, plot_qinit_only=True, zero_ones_labels=zero_ones_labels)
            tb_writer.add_figure(f'gammas/train_init_gammas', plt.gcf(), 0)
            plot_files(train_dset, val_file_indices, plot_qinit_only=True, zero_ones_labels=zero_ones_labels)
            tb_writer.add_figure(f'gammas/val_init_gammas', plt.gcf(), 0)

        # Save starting point model.
        save_path = tb_writer.log_dir
        if not os.path.exists(f'{save_path}/checkpoints'):
            os.mkdir(f'{save_path}/checkpoints')

        if args.run_dist:
            ddp_model.module.save_model(f'{save_path}/checkpoints/model_ep_0.pt', 0, 0)
        else:
            model.save_model(f'{save_path}/checkpoints/model_ep_0.pt', 0, 0)

    if args.run_dist:
        dist.barrier()

    for ep in range(starting_epoch, starting_epoch + args.epochs):
        if args.run_dist:
            train_dsampler.set_epoch(ep)

        ep_losses = []
        ep_raw_losses = []
        for batch in train_dl:
            step_counter += 1

            batch_losses = []
            batch_raw_losses = []
            for _, _, _, x, qinit, y in batch:
                pi_init = torch.ones(qinit.size()[1],
                                     dtype=torch.double,
                                     requires_grad=False) / qinit.size()[1]

                if args.run_dist:
                    ddp_model.module.train()
                    log_gamma, gamma, pi, Li = ddp_model(x, pi=pi_init, gamma=qinit)
                else:
                    model.train()
                    log_gamma, gamma, pi, Li = model(x, pi=pi_init, gamma=qinit)

                loss, raw_loss = model.compute_loss(y)
                batch_losses.append(loss)
                batch_raw_losses.append(raw_loss)
                ep_losses.append(loss.detach())
                ep_raw_losses.append(raw_loss)

            avg_loss = sum(batch_losses) / len(batch_losses)
            avg_raw_loss = sum(batch_raw_losses) / len(batch_raw_losses)
            model.optimize_params(avg_loss)

            if rank == 0:
                model.tb_log_params(tb_writer, 'params', 'step', step_counter)
                model.tb_log_params_grads(tb_writer, 'grads', 'step', step_counter)

                if args.run_dist:
                    KL_Ew, KL_Eb = ddp_model.module.compute_KL()
                else:
                    KL_Ew, KL_Eb = model.compute_KL()

                tb_writer.add_scalar('params/KL_Eb_step', KL_Eb.item(), step_counter)
                tb_writer.add_scalar('params/KL_Ew_step', KL_Ew.item(), step_counter)

            if args.run_dist:
                batch_avg_losses = [0 for _ in range(world_size)]
                batch_avg_raw_losses = [0 for _ in range(world_size)]
                dist.all_gather_object(batch_avg_losses, avg_loss.item())
                dist.all_gather_object(batch_avg_raw_losses, avg_raw_loss.item())
            else:
                batch_avg_losses = [avg_loss]
                batch_avg_raw_losses = [avg_raw_loss]

            if rank == 0:
                batch_avg_loss = sum(batch_avg_losses) / len(batch_avg_losses)
                batch_avg_raw_loss = sum(batch_avg_raw_losses) / len(batch_avg_raw_losses)
                print(f"BatchLoss: {sum(batch_avg_losses) / len(batch_avg_losses)}")
                tb_writer.add_scalar('train/loss_step', batch_avg_loss, step_counter)
                tb_writer.add_scalar('train/raw_loss_step', batch_avg_raw_loss, step_counter)

            if step_counter % val_steps == 0:
                if args.run_dist:
                    ddp_model.module.eval()
                else:
                    model.eval()

                val_ep = step_counter // val_steps
                val_losses = []
                val_raw_losses = []
                with torch.no_grad():
                    for batch in val_dl:
                        for _, _, _, x, qinit, y in batch:
                            pi_init = torch.ones(qinit.size()[1],
                                                 dtype=torch.double,
                                                 requires_grad=False) / qinit.size()[1]

                            if args.run_dist:
                                log_gamma, gamma, pi, Li = ddp_model(x, pi=pi_init, gamma=qinit)
                            else:
                                log_gamma, gamma, pi, Li = model(x, pi=pi_init, gamma=qinit)

                            loss, raw_loss = model.compute_loss(y)
                            val_losses.append(loss)
                            val_raw_losses.append(raw_loss)

                if args.run_dist:
                    dist.barrier()

                if args.run_dist:
                    val_avg_losses = [None for _ in range(world_size)]
                    val_avg_raw_losses = [None for _ in range(world_size)]
                    dist.all_gather_object(val_avg_losses, torch.mean(torch.stack(val_losses)))
                    dist.all_gather_object(val_avg_raw_losses, torch.mean(torch.stack(val_raw_losses)))
                else:
                    val_avg_losses = [torch.mean(torch.stack(val_losses))]
                    val_avg_raw_losses = [torch.mean(torch.stack(val_raw_losses))]

                if args.run_dist:
                    train_der = eval_model(ddp_model, train_dl, args.run_dist, der_train, rank, world_size)
                    val_der = eval_model(ddp_model, val_dl, args.run_dist, der_val, rank, world_size)

                    if args.in_eval_gt_rttm_dir:
                        eval_der = eval_model(ddp_model, eval_dl, args.run_dist, der_eval, rank, world_size)
                else:
                    train_der = eval_model(model, train_dl, args.run_dist, der_train, rank, world_size)
                    val_der = eval_model(model, val_dl, args.run_dist, der_val, rank, world_size)

                    if args.in_eval_gt_rttm_dir:
                        eval_der = eval_model(model, eval_dl, args.run_dist, der_eval, rank, world_size)

                if rank == 0:
                    # ToDo: Mention this val_ep conting into README !!! It can confuse a new user once eval after ..
                    #   flag is set.
                    tb_writer.add_scalar('der/train_ep', train_der, val_ep)
                    tb_writer.add_scalar('der/val_ep', val_der, val_ep)

                    if args.in_eval_gt_rttm_dir:
                        tb_writer.add_scalar('der/eval_ep', eval_der, val_ep)

                    if args.plot_gammas:
                        plot_files(train_dset, train_file_indices, zero_ones_labels=zero_ones_labels)
                        tb_writer.add_figure(f'gammas/train_ep', plt.gcf(), val_ep)
                        plot_files(train_dset, train_file_indices, processed=True, zero_ones_labels=zero_ones_labels)
                        tb_writer.add_figure(f'gammas/train_loss_inp_ep', plt.gcf(), val_ep)

                        plot_files(val_dset, val_file_indices, zero_ones_labels=zero_ones_labels)
                        tb_writer.add_figure(f'gammas/val_ep', plt.gcf(), val_ep)
                        plot_files(val_dset, val_file_indices, processed=True, zero_ones_labels=zero_ones_labels)
                        tb_writer.add_figure(f'gammas/val_loss_inp_ep', plt.gcf(), val_ep)

                    avg_val_loss = sum(val_avg_losses) / len(val_avg_losses)
                    avg_val_raw_loss = sum(val_avg_raw_losses) / len(val_avg_raw_losses)

                    print(f"AvgValLoss: {avg_val_loss}")

                    tb_writer.add_scalar('train/val_loss_epoch', avg_val_loss, val_ep)
                    tb_writer.add_scalar('train/val_raw_loss_epoch', avg_val_raw_loss, val_ep)

            if rank == 0 and step_counter % save_steps == 0:
                save_ep = step_counter // save_steps
                save_path = tb_writer.log_dir
                if not os.path.exists(f'{save_path}/checkpoints'):
                    os.mkdir(f'{save_path}/checkpoints')

                if args.run_dist:
                    ddp_model.module.save_model(f'{save_path}/checkpoints/model_ep_{save_ep}.pt', save_ep,
                                                step_counter)
                else:
                    model.save_model(f'{save_path}/checkpoints/model_ep_{save_ep}.pt', save_ep, step_counter)

        # Log training progress
        if args.run_dist:
            ep_avg_losses = [None for _ in range(world_size)]
            ep_avg_raw_losses = [None for _ in range(world_size)]
            dist.all_gather_object(ep_avg_losses, torch.mean(torch.stack(ep_losses)))
            dist.all_gather_object(ep_avg_raw_losses, torch.mean(torch.stack(ep_raw_losses)))
        else:
            ep_avg_losses = [torch.mean(torch.stack(ep_losses))]
            ep_avg_raw_losses = [torch.mean(torch.stack(ep_raw_losses))]

        if rank == 0:
            avg_ep_loss = sum(ep_avg_losses) / len(ep_avg_losses)
            avg_ep_raw_loss = sum(ep_avg_raw_losses) / len(ep_avg_raw_losses)
            print(f'AvgEpLoss: {avg_ep_loss}')

            tb_writer.add_scalar('train/loss_epoch', avg_ep_loss, ep)
            tb_writer.add_scalar('train/raw_loss_epoch', avg_ep_raw_loss, ep)

        if args.run_dist:
            dist.barrier()
    print("DONE")

    if rank == 0:
        tb_writer.close()

    if args.run_dist:
        dist.destroy_process_group()
