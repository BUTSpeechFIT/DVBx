#!/usr/bin/env python3

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

from scipy.optimize import linear_sum_assignment
import torch

from VBx.losses.loss import Loss
from VBx.enums.gamma_types import GammaTypes


class EDELoss(Loss):
    """
    This loss is calculated as (fa + miss) / recording_length, which does not represent the DER metric as we know it,
    but rather Expected Detection Error Rate.
    """

    def __init__(self, use_scale=False, log_loss_scale=None, use_logprobs=True, use_adaptive_scale=False,
                 max_gamma_val=None):
        super().__init__(use_scale, log_loss_scale, use_logprobs, use_adaptive_scale, max_gamma_val,
                         need_gamma_type=GammaTypes.PROB)

    def __call__(self, decisions: torch.Tensor, target: torch.Tensor):
        """
        Inputs:
            decisions - TxS tensor (T - time, S - number of speakers) representing decisions made by a model
            target    - TxS tensor (T - time, S - number of speakers) representing ground truth
        """
        # Processed VBx responsibilities (probabilities).
        decisions = super()._process_decisions(decisions)
        decisions_t = torch.transpose(decisions, 0, 1)

        cost_mx2 = torch.matmul(1 - decisions_t, target) + torch.matmul(decisions_t, 1 - target)
        pred_alig, ref_alig = linear_sum_assignment(cost_mx2.detach())

        return torch.sum(cost_mx2[(pred_alig, ref_alig)]) / target.size()[0]
