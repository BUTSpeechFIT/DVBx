#!/usr/bin/env python3

# @Authors: Lukas Burget (burget@fit.vutbr.cz), Brno University of Technology

import torch


def logprobs_to_logodds(logprobs):
    """ Transform a matrix of multi-class (unnormalized) log probabilities into (binary) log odds (logits).
    Input:  logprobs - TxC matrix, where T is number of examples and C is number of classes
    Output: logodds  - TxC matrix
    """
    C = logprobs.shape[1]
    logodds = torch.zeros_like(logprobs)

    for m in range(C):
        impostor_indices = torch.where(torch.arange(C) != m)
        logodds[:, m] = logprobs[:, m] - torch.logsumexp(logprobs.T[impostor_indices], dim=0)

    return logodds
