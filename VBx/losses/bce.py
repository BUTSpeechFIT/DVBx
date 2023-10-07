#!/usr/bin/env python3

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torch.nn.functional import logsigmoid
import numpy as np

from VBx.losses.loss import Loss
from VBx.enums.gamma_types import GammaTypes


class BCELoss(Loss):
    def __init__(self, use_scale=False, log_loss_scale=None, use_logprobs=True, use_adaptive_scale=False,
                 max_gamma_val=None):
        super().__init__(use_scale, log_loss_scale, use_logprobs, use_adaptive_scale, max_gamma_val,
                         need_gamma_type=GammaTypes.LOGIT)

    def __call__(self, decisions: torch.Tensor, target: torch.Tensor):
        """
        Inputs:
            decisions - TxS tensor (T - time, S - number of speakers) representing decisions made by a model
            target    - TxS tensor (T - time, S - number of speakers) representing ground truth
        """
        logits = super()._process_decisions(decisions)
        logits_t = logits.detach().transpose(0, 1)

        cost_mx = -logsigmoid(logits_t.unsqueeze(0)).bmm(target.unsqueeze(0)) - \
                  logsigmoid(-logits_t.unsqueeze(0)).bmm(1 - target.unsqueeze(0))
        pred_alig, ref_alig = linear_sum_assignment(cost_mx[0].to("cpu"))
        assert (np.all(pred_alig == np.arange(decisions.shape[-1])))
        target = target[:, ref_alig]

        # ToDo: Targets may have -1 values in the future if the conversations get padded for another batching approach.
        activation_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        return torch.mean(activation_loss)
