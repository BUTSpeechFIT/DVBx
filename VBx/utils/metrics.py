#!/usr/bin/env python3

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.loader import RTTMLoader
import torch
import numpy as np

class DER:
    def __init__(self, file_list, gt_rttm_path):
        self.rttms = dict()
        for f in file_list:
            self.rttms[f] = RTTMLoader(f'{gt_rttm_path}/{f}.rttm').loaded_[f]

    def der_tensor(self, start, end, decisions: torch.tensor, file_name):
        """
        Compute DER between the decisions and the reference torch tensors
        with dimension TxN, where T is the time axis and N is the max number of speakers.

        Inputs:
            start     - ...
            end       - ...
            decisions - TxN torch tensor containing binary decisions.
            reference - TxN torch tensor containing binary labels.
        Returns:
            Tuple containing the following elements:
            (DER score between the decisions and the reference labels,
             Speech amount in the given file).
        """

        annot: Annotation = DER._transfer_binary_decisions_to_pyannote_annotation(start, end, decisions)

        speech_amount = 0
        for seg, _ in self.rttms[file_name].itertracks():
            speech_amount += seg.duration

        return DiarizationErrorRate()(self.rttms[file_name], annot), speech_amount

    @staticmethod
    def _transfer_binary_decisions_to_pyannote_annotation(start, end, decisions) -> Annotation:
        starts, ends, out_labels = DER._merge_adjacent_labels(start, end, decisions)

        annotation_sys = Annotation()

        for s, e, l in zip(starts, ends, out_labels):
            s = Segment(s, e)
            annotation_sys[s, annotation_sys.new_track(s)] = l

        return annotation_sys

    @staticmethod
    def _merge_adjacent_labels(starts, ends, labels):
        """ Labeled segments defined as start and end times are compacted in such a way that
        adjacent or overlapping segments with the same label are merged. Overlapping
        segments with different labels are further adjusted not to overlap (the boundary
        is set in the middle of the original overlap).
        Input:
             starts - array of segment start times in seconds
             ends   - array of segment end times in seconds
             labels - array of segment labels (of any type)
        Outputs:
              starts, ends, labels - compacted and ajusted version of the input arrays
        """
        # Merge neighbouring (or overlaping) segments with the same label
        adjacent_or_overlap = np.logical_or(np.isclose(ends[:-1], starts[1:]), ends[:-1] > starts[1:])
        to_split = np.nonzero(np.logical_or(~adjacent_or_overlap, labels[1:] != labels[:-1]))[0]
        starts = starts[np.r_[0, to_split + 1]]
        ends = ends[np.r_[to_split, -1]]
        labels = labels[np.r_[0, to_split + 1]]

        # Fix starts and ends times for overlapping segments
        overlaping = np.nonzero(starts[1:] < ends[:-1])[0]
        ends[overlaping] = starts[overlaping + 1] = (ends[overlaping] + starts[overlaping + 1]) / 2.0
        return starts, ends, labels
