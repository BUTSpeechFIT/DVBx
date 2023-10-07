#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
import os
from collections import OrderedDict
from typing import List, Tuple


def hard_labels_to_rttm(
    matrix: np.ndarray,
    spk_labels: List[str],
    utt_id: str,
    rttm_path: str,
    precision: float
):
    """
        reads an NfxNs matrix containing hard speaker activities (labels 1/0)
        at the given precision. The speaker labels should correspond to the
        order in the array and are used to assign the speech segments in the
        rttm. The utterance ID is used for the second field of each rttm row.
    """
    matrix_extended = np.concatenate((
        np.zeros((1, matrix.shape[1])),
        matrix,
        np.zeros((1, matrix.shape[1]))), axis=0)
    changes_dict = {}
    for s in range(len(spk_labels)):
        spk_changes = np.where(matrix_extended[1:, s] - matrix_extended[:-1, s]
                               )[0].astype(float)
        if spk_changes.shape[0] > 0:
            if spk_changes[-1] == matrix.shape[0]:
                spk_changes[-1] -= 1  # avoid reading out of array
            beg = spk_changes[:-1]
            end = spk_changes[1:]
            # So far, beg and end include the silences in between
            beg = beg[::2]
            end = end[::2]
            assert beg.shape[0] == end.shape[0], "Amount of beginning and \
                                               end of segments do not match."
            for pos in range(beg.shape[0]):
                time_beg = beg[pos] / precision
                time_length = (end[pos] - beg[pos]) / precision
                changes_dict[(time_beg, s)] = f"SPEAKER {utt_id} 1 {time_beg} {time_length} <NA> <NA> {spk_labels[s]} <NA> <NA>\n"
    changes_dict = OrderedDict(sorted(changes_dict.items()))
    if not os.path.exists(os.path.dirname(rttm_path)):
        os.makedirs(os.path.dirname(rttm_path))
    with open(rttm_path, 'w') as f:
        for k, v in changes_dict.items():
            f.write(v)


def rttm_to_hard_labels(
    rttm_path: str,
    precision: float,
    length: float = -1
) -> Tuple[np.ndarray, List[str]]:
    """
        reads the rttm and returns a NfxNs matrix encoding the segments in
        which each speaker is present (labels 1/0) at the given precision.
        Ns is the number of speakers and Nf is the resulting number of frames,
        according to the parameters given.
        Nf might be shorter than the real length of the utterance, as final
        silence parts cannot be recovered from the rttm.
        If length is defined (s), it is to account for that extra silence.
        In case of silence all speakers are labeled with 0.
        In case of overlap all speakers involved are marked with 1.
        The function assumes that the rttm only contains speaker turns (no
        silence segments).
        The overlaps are extracted from the speaker turn collisions.
    """
    # each row is a turn, columns denote beginning (s) and duration (s) of turn
    data = np.loadtxt(rttm_path, usecols=[3, 4])
    # speaker id of each turn
    spks = np.loadtxt(rttm_path, usecols=[7], dtype='str')
    spk_ids = np.unique(spks)
    Ns = len(spk_ids)
    if data.shape[0] == 2 and len(data.shape) < 2:  # if only one segment
        data = np.asarray([data])
        spks = np.asarray([spks])
    # length of the file (s) that can be recovered from the rttm,
    # there might be extra silence at the end
    #len_file = data[-1][0]+data[-1][1] #last item in the rttm might not be the one containing the last segment of speech
    len_file = np.max(np.sum(data,axis=1)) 
    if length > len_file:
        len_file = length

    # matrix in given precision
    matrix = np.zeros([int(round(len_file*precision)), Ns])
    # ranges to mark each turn
    ranges = np.around((np.array([data[:, 0],
                        data[:, 0]+data[:, 1]]).T*precision)).astype(int)

    for s in range(Ns):  # loop over speakers
        # loop all the turns of the speaker
        for init_end in ranges[spks == spk_ids[s], :]:
            matrix[init_end[0]:init_end[1], s] = 1  # mark the spk
    return matrix, spk_ids
