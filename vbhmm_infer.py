#!/usr/bin/env python

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

import argparse
import os
import numpy as np
from scipy.linalg import eigh
import torch
import torch.distributed as dist

from VBx.diarization_lib import merge_adjacent_labels, mkdir_p
from VBx.utils.kaldi_utils import read_plda
from VBx.models import VBx as VBx
from VBx.dataset.dataset import BasicDataset
from VBx.dataset.data_file_loader import DataFileLoader
from VBx.utils.config import process_args
from VBx.utils.metrics import DER


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


def write_output(fp, file_name, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
                 f'<NA> <NA> {label + 1} <NA> <NA>{os.linesep}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in-file-list', required=False, type=str,
                        help='txt list of files')
    parser.add_argument('--out-rttm-dir', required=True, type=str,
                        help='Directory to store output rttm files')
    parser.add_argument('--xvec-ark-dir', required=True, type=str,
                        help='Kaldi ark file with x-vectors from one or more '
                             'input recordings. Attention: all x-vectors from '
                             'one recording must be in one ark file')
    parser.add_argument('--segments-dir', required=True, type=str,
                        help='File with x-vector timing info. See '
                             'diarization_lib.read_xvector_timing_dict')
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
    parser.add_argument('--target-energy', required=False, type=float,
                        default=1.0,
                        help='Parameter affecting AHC if the similarity '
                             'matrix is obtained with PLDA. See '
                             'diarization_lib.kaldi_ivector_plda_scoring_dense')
    parser.add_argument('--in-INITlabels-dir', required=False, type=str,
                        help='Directory with the AHC initialization xvector labels. '
                             'If not provided, the system will apply AHC first.')
    parser.add_argument('--model-path', required=True, type=str, help="Path to saved .pt model checkpoint.")
    parser.add_argument('--use-gmm', required=False, default=False, action='store_true',
                        help='If present, GMM is used instead of HMM, i.e. ploop=0.')
    parser.add_argument('--output-2nd', type=bool, default=False,
                        help='If present, system will also output the second most probable speaker.')

    # Default config. Those parameters don't have to be set.
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
    parser.add_argument('--lr', default=1e-3, help='Learning rates.')
    parser.add_argument('--max-iters', type=int, default=10,
                        help='Max number of VB iterations during training (if early_stop_vb is False, than it\'')
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

    process_args(args)

    plda_mu, plda_psi, plda_tr = read_kaldi_plda(args.plda_file)

    model = VBx.DiscriminativeVB(device=torch.device("cpu"),
                                 phi=torch.from_numpy(plda_psi[:args.lda_dim].copy()),
                                 tr=torch.from_numpy(plda_tr.copy()),
                                 # fa,fb, loop_p, init_smoothing are loaded later on from .pt checkpoint.
                                 fa=1,
                                 fb=1,
                                 loop_p=0.5,
                                 config=args,
                                 init_smoothing=7,
                                 epsilon=1e-6,
                                 use_gmm=args.use_gmm)
    model.load_model(args.model_path)

    data_list = np.loadtxt(args.in_file_list, dtype=object)
    dfl = DataFileLoader(plda_mu,
                         args.segments_dir,
                         args.in_INITlabels_dir,
                         args.xvec_ark_dir,
                         None,
                         args.xvec_transform,
                         7,
                         ahc_threshold=args.threshold,
                         eval_mode=True)
    dset = BasicDataset(data_list, dfl)

    model.eval()
    with torch.no_grad():
        for segs_dict, seg_names, file_name, x, qinit, y in dset:
            pi_init = torch.ones(qinit.size()[1],
                                 dtype=torch.double,
                                 requires_grad=False) / qinit.size()[1]

            log_gamma, gamma, pi, Li = model(x, pi=pi_init, gamma=qinit)
            labels1st = torch.argsort(-gamma, dim=1)[:, 0].detach().numpy()
            labels2nd = np.argsort(-gamma, axis=1)[:, 1].detach().numpy()

            start, end = segs_dict[file_name][1].T

            starts, ends, out_labels = merge_adjacent_labels(start, end, labels1st)
            mkdir_p(args.out_rttm_dir)
            with open(os.path.join(args.out_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
                write_output(fp, file_name, out_labels, starts, ends)

            if args.output_2nd and args.init.endswith('VB') and gamma.shape[1] > 1:
                starts, ends, out_labels2 = merge_adjacent_labels(start, end, labels2nd)
                output_rttm_dir = f'{args.out_rttm_dir}2nd'
                mkdir_p(output_rttm_dir)
                with open(os.path.join(output_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
                    write_output(fp, file_name, out_labels2, starts, ends)
