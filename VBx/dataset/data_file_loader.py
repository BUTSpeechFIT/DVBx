#!/usr/bin/env python3

"""
@Authors: Dominik Klement, Mireia Diez, Federico Landini (xkleme15@vutbr.cz, mireia@fit.vutbr.cz, landini@fit.vutbr.cz),
Brno University of Technology
"""

from VBx.diarization_lib import l2_norm, cos_similarity, twoGMMcalib_lin
from kaldi_io import read_vec_flt_ark
from itertools import groupby
import h5py
import numpy as np
from torch import from_numpy as torch_from_numpy
import fastcluster
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform


class DataFileLoader:
    def __init__(self,
                 plda_mu,
                 segments_dir_path,
                 init_labels_dir_path,
                 xvec_ark_dir_path,
                 gt_labels_dir_path,
                 xvec_transform_path,
                 init_smoothing,
                 gt_label_type='PROBABILITIES',
                 ahc_threshold=0.015,
                 eval_mode=False):
        self.plda_mu = plda_mu
        self.segments_dir_path = segments_dir_path
        self.init_labels_dir_path = init_labels_dir_path
        self.xvec_ark_dir_path = xvec_ark_dir_path
        self.gt_labels_dir_path = gt_labels_dir_path
        self.xvec_transform_path = xvec_transform_path
        self.init_smoothing = init_smoothing
        self.gt_label_type = gt_label_type
        self.ahc_threshold = ahc_threshold

        # If true, we only need the xvectors and segments, and init labels, no gt labels
        self.eval_mode = eval_mode

        with h5py.File(xvec_transform_path, 'r') as f:
            self.mean1 = np.array(f['mean1'])
            self.mean2 = np.array(f['mean2'])
            self.lda = np.array(f['lda'])

    def load_file(self, file_name):
        arkit = read_vec_flt_ark(self.xvec_ark_dir_path + file_name + ".ark")
        recit = groupby(arkit, lambda e: e[0].rsplit('_', 1)[0])

        fn, segs = next(iter(recit))
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)
        x = l2_norm(self.lda.T.dot((l2_norm(x - self.mean1)).transpose()).transpose() - self.mean2)

        if self.init_labels_dir_path is None:
            # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise
            # similarities between all x-vectors)
            scr_mx = cos_similarity(x)
            # Figure out utterance specific args.threshold for AHC
            thr, _ = twoGMMcalib_lin(scr_mx.ravel())
            # output "labels" is an integer vector of speaker (cluster) ids
            scr_mx = squareform(-scr_mx, checks=False)
            lin_mat = fastcluster.linkage(
                scr_mx, method='average', preserve_input='False')
            del scr_mx
            adjust = abs(lin_mat[:, 2].min())
            lin_mat[:, 2] += adjust
            labels1st = fcluster(lin_mat, -(thr + self.ahc_threshold) + adjust, criterion='distance') - 1
        else:
            labels1st = np.loadtxt(self.init_labels_dir_path + file_name + ".init_labs", dtype=int)

        if self.eval_mode:
            max_label = np.max(labels1st)
            y_labels = None
        else:
            y_labels_N = np.loadtxt(self.gt_labels_dir_path + file_name + ".init_labs")
            if len(y_labels_N.shape) == 1:
                y_labels_N = np.expand_dims(y_labels_N, 1)

            max_label = np.maximum(np.shape(y_labels_N)[1] - 1, np.max(labels1st)).astype(int)

            if self.gt_label_type == 'SINGLE_SPK':
                """
                The i-th frame label is a one-hot encoding of the prevalent speaker in the frame 
                (i.e. speaking for most of the time).
                """
                y_labels_N = np.argmax(y_labels_N, axis=1)
            elif self.gt_label_type == 'ZERO_ONES':
                """
                The i-th frame label is the binary encoding of the presence of the speakers in the given frame
                (i.e. if a speaker s speaks in the given frame, the corresponding label is equal to 1).
                """
                y_labels_N = (y_labels_N > 0).astype(int)
            elif self.gt_label_type == 'PROBABILITIES':
                """
                The i-th frame label is proportional to the amount of speaker's speech in the frame
                (e.g. we have a 1.5s frame and 1s speaks 1st speaker and 0.5s speaks 2nd one, the corresponding labels
                are: (2/3, 1/3)).
                """
                y_labels = y_labels_N
            else:
                raise Exception("Ground truth label type not implemented")

            y_labels = np.zeros((len(y_labels_N), max_label + 1))
            if self.gt_label_type == 'SINGLE_SPK':
                y_labels[range(len(y_labels_N)), y_labels_N] = 1.0
            else:
                y_labels[:y_labels_N.shape[0], :y_labels_N.shape[1]] = y_labels_N

        qinit = np.zeros((len(labels1st), max_label + 1))
        qinit[range(len(labels1st)), labels1st] = 1.0
        fea = (x - self.plda_mu)

        return (
            self._read_xvector_timing_dict(self.segments_dir_path + file_name),
            seg_names,
            file_name,
            torch_from_numpy(fea),
            torch_from_numpy(qinit),
            np.empty(shape=(1, 1)) if self.eval_mode else torch_from_numpy(y_labels)
        )

    @staticmethod
    def _read_xvector_timing_dict(kaldi_segments):
        """ Loads kaldi 'segments' file with the timing information for individual x-vectors.
        Each line of the 'segments' file is expected to contain the following fields:
        - x-vector name (which needs to much the names of x-vectors loaded from kaldi archive)
        - file name of the recording from which the xvector is extracted
        - start time
        - end time
        Input:
            kaldi_segments - path (including file name) to the Kaldi 'segments' file
        Outputs:
             segs_dict[recording_file_name] = (array_of_xvector_names, array_of_start_and_end_times)
        """
        segs = np.loadtxt(kaldi_segments, dtype=object)
        split_by_filename = np.nonzero(segs[1:, 1] != segs[:-1, 1])[0] + 1
        return {s[0, 1]: (s[:, 0], s[:, 2:].astype(float)) for s in np.split(segs, split_by_filename)}
