#!/usr/bin/env python3

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

from torch import log as torch_log, exp as torch_exp
from torch.nn.functional import softmax, log_softmax
import torch

from VBx.enums.gamma_types import GammaTypes
from VBx.utils.utils import logprobs_to_logodds


class Loss:
    def __init__(self, use_scale=False, log_loss_scale=None, use_logprobs=True, use_adaptive_scale=False,
                 max_gamma_val=0.95, need_gamma_type=GammaTypes.PROB):
        self.use_scale = use_scale
        self.use_logprobs = use_logprobs
        self.use_adaptive_scale = use_adaptive_scale
        self.max_gamma_val = max_gamma_val
        self.need_gamma_type = need_gamma_type

        if use_scale and log_loss_scale is None:
            raise 'loss_scale parameter cannot be None if use_scale is true.'

        self.log_loss_scale = log_loss_scale

    # ToDo: Merge scale_decisions and process_decisions because we may produce numerical imprecisions in some
    #  settings of "use_logits" and "need_logits" parameters.
    def _scale_decisions(self, decisions):
        # if use_logprobs is true then decisions are log responsibilities
        logprobs = decisions if self.use_logprobs else torch_log(decisions)
        scaled_logprobs = logprobs * torch.exp(self.log_loss_scale)

        if self.use_logprobs:
            return log_softmax(scaled_logprobs, dim=1)

        return softmax(scaled_logprobs, dim=1)

    def _process_decisions(self, decisions):
        if self.use_scale:
            decisions = self._scale_decisions(decisions)

        if self.need_gamma_type == GammaTypes.PROB:
            # The loss needs probabilities but decisions are log probabilities.
            if self.use_logprobs:
                decisions = torch_exp(decisions)
        elif self.need_gamma_type == GammaTypes.LOG_PROB:
            # The loss needs log probabilities but decisions are probabilities.
            if not self.use_logprobs:
                decisions = torch_log(decisions)
        else:
            # gamma_type = LOGIT
            decisions = logprobs_to_logodds(decisions if self.use_logprobs else torch_log(decisions))

        return decisions
