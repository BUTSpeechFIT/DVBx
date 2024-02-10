# Discriminative Training of VBx Diarization (DVBx)

Training and evaluation recepie for DVBx (an extension of previously published [VBx diarization model](https://github.com/BUTSpeechFIT/VBx)).

It contains:

- Data preparation (x-vector extraction, AHC initial labels and ground truth labels generation)
- Training recipe
- Inference recipe

### Setup

Create a conda environment by running the following command:

```bash
conda env create -f conda_env.yml
```

Activate the new environment:

```bash
conda activate dvbx
```

To prepare the training set, e.g. the set of xvectors, the initialization labels obtained from AHC and the ground truth
labels, use:

    ./prepare_train_set.sh xvectors experiment_dir xvector_dir

    ./prepare_train_set.sh diarization experiment_dir xvector_dir

The `prepare_train_set.sh` script contains a few variables that have to be changed:
```bash
model_type="8k" # Either 8k or 16k
WAV_DIR= # wav files directory
LAB_DIR= # lab files directory with VAD segments
REF_DIR= # reference rttm files directory
FILE_LIST= # txt list of files to process
```
- `model_type` can be set either to `8k` or `16k` - to get the best performance, set the type according to your dataset sampling frequency (`8k` model works on `16k` data and vice-versa with suboptimal performance)
- `WAV_DIR` contains a path (recommend using absolute paths) to a directory with `.wav` files.
- `LAB_DIR` contains a path to a directory with VAD segments (i.e. a text file corresponding to KALDI segments format)
  - An example of a single line representing a single VAD speech segment: `0.130	4.010	speech`
- `REF_DIR` contains a path to a directory with reference RTTM files
- `FILE_LIST` is a path to a text file containing the names of files to process

The `process_train_set.sh` loops through all lines in `FILE_LIST` and looks at the following paths: `${WAV_DIR}/${fn}.wav`, `${LAB_DIR}/${fn}.lab`, `${REF_DIR}/${fn}.rttm`, where `${fn}` denotes a single line in `FILE_LIST`.
Therefore, all the files (wav, VAD segments, RTTMs) need to have the same name (i.e. `iaaa.wav`, `iaaa.lab`, `iaaa.rttm`) and `FILE_LIST` must contain a single name per line (without any extensions).
Only the files listed in the `FILE_LIST` will be processed!

You will obtain the GT and the AHC for the xvectors in the directories:

    $exp_dir/xvector_GT_labels

    $exp_dir/out_dir_AHC

### Config

Example of a config file can be found in `config/config.yml`

| Parameter                |            Default Value             |     Type     | Description                                                                                                                                                                                                                                                                                            |
|--------------------------|:------------------------------------:|:------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| avg_last_n_iters         |                  -1                  |     Int      | Number of last VB iterations that are averaged during gradient computation (if backprop_type = after_each_iter). All if avg_last_n_iters = -1.                                                                                                                                                         |                                                                                                                                                         
| backprop_type            |           after_each_iter            |    String    | Describes the way of computing gradients: <ul> <li>after_each_iter - computes the loss in after VB iteration, then averages the losses (if use_loss_weights is true then the avg is weighted)</li> <li>after_convergence - computes the loss after VB converges (or max_iters limit is hit)</li> </ul> |
| batch_size               |                  8                   |     Int      | Number of samples to estimate the gradients from.                                                                                                                                                                                                                                                      |
| early_stop_vb            |                False                 |     Bool     | If true, the algorithm stops if the ELBO stops improving.                                                                                                                                                                                                                                              |
| eval_max_iters           |                  40                  |     Int      | Maximum number of VB iterations during evaluation.                                                                                                                                                                                                                                                     |
| eval_early_stop_vb       |                 True                 |     Bool     | Same as early_stop_vb during evaluation.                                                                                                                                                                                                                                                               |
| epochs                   |                 500                  |     Int      | Number of loops through the whole dataset.                                                                                                                                                                                                                                                             |
| initial_loss_scale       |                  1                   |    Float     | Initial value of loss scale.                                                                                                                                                                                                                                                                           |
| loss                     |                 EDE                  |    String    | BCE/EDE.                                                                                                                                                                                                                                                                                               |
| lr                       |                 None                 |  Int/Object  | If set to a number, all trainable parameters will be trained using the same learning rate. If set to an object, each key specifies a set of comma-separated parameters for which a specific learning rate is set Key other specifies all not specified parameters.                                     |
| max_iters                |                  10                  |     Int      | Maximum number of VB iterations.                                                                                                                                                                                                                                                                       |
| regularization_coeff_eb  |                  0                   |    Float     | Between-class PLDA covariance matrix KL divergence regularization loss weight.                                                                                                                                                                                                                         |
| regularization_coeff_ew  |                  0                   |    Float     | Within-class PLDA covariance matrix KL divergence regularization loss weight.                                                                                                                                                                                                                          |
| trainable_parameters     | Fa, Fb, loop_prob, initial smoothing | List[String] | List of parameters that will be trained.                                                                                                                                                                                                                                                               |
| use_full_tr              |                False                 |     Bool     | If true, full transition matrix is used.                                                                                                                                                                                                                                                               |
| use_loss_scale           |                False                 |     Bool     | If true, log probabilities are scaled before being passed into a loss function.                                                                                                                                                                                                                        |
| use_loss_weights         |                False                 |     Bool     | If true, backprop_type is set to "after_each_iter" the losses after each VB iteration are summed and weighted by the weights (initially all set to ones).                                                                                                                                              |
| use_regularization       |                False                 |     Bool     | If true, KL divergence regularization for PLDA matrices is added to the loss.                                                                                                                                                                                                                          |
| use_sigmoid_loss_weights |                False                 |     Bool     | If true, sigmoid loss weights are used instead of softmax ones (see in code, models/VBx.py)                                                                                                                                                                                                            |

### Training

The training script can be run in two different modes: single-threaded, distributed (parallel).
You can find an example of how to run the distributed training below. For the explanation of how the torchrun works,
please visit: https://pytorch.org/docs/stable/elastic/run.html.

In order to run the training on a single thread only, simply run the `vbhmm_train.py` as a python script and make sure
the `--run-dist` option is **NOT** present.

Adjust the paths and use the following command to run the training process:

```shell
torchrun --nnodes 1
         --nproc_per_node 8
         vbhmm_train.py
         --config-path config/config.yml
         --eval-segments-dir xvector_dir/segments/
         --eval-xvec-ark-dir xvector_dir/xvectors/
         --exp-name-tag my_tag
         --Fa 1
         --Fb 1
         --gt-label-type PROBABILITIES
         --in-eval-gt-rttm-dir experiment_dir/ref_rttms
         --in-eval-GTlabels-dir xvector_dir/xvector_GT_labels/
         --in-eval-INITlabels-dir experiment_dir/out_dir_AHC/
         --in-eval-list data_lists/callhome_part2.txt
         --in-GTlabels-dir xvector_dir/xvector_GT_labels/
         --in-INITlabels-dir experiment_dir/out_dir_AHC/
         --in-train-gt-rttm-dir experiment_dir/ref_rttms
         --in-trainlist data_lists/callhome_p1_train_half.txt
         --in-val-gt-rttm-dir experiment_dir/ref_rttms
         --in-val-GTlabels-dir xvector_dir/xvector_GT_labels
         --in-val-INITlabels-dir experiment_dir/out_dir_AHC
         --in-vallist data_lists/callhome_p1_val_half.txt
         --lda-dim 128
         --loopP 0.5
         --plda-file VBx/models/ResNet101_8kHz/plda
         --segments-dir xvector_dir/segments/
         --threshold -0.015
         --use-gmm
         --val-segments-dir xvector_dir/segments
         --val-xvec-ark-dir xvector_dir/xvectors/
         --xvec-ark-dir xvector_dir/xvectors/
         --xvec-transform VBx/models/ResNet101_8kHz/transform.h5
         --run-dist
```

##### Parameters description

| Argument                    |  Type  | Description                                                                                                                            |
|-----------------------------|:------:|----------------------------------------------------------------------------------------------------------------------------------------|
| config-path                 | String | Path to a configuration YAML file.                                                                                                     |
| continue-log-dir            | String | Path to tensorboard log directory to continue the training process from the last checkpoint.                                           |
| eval-after-epochs           |  Int   | Number of epochs after the model is evaluated.                                                                                         |
| eval-after-steps            |  Int   | Number of steps after the model is evaluated (if set, eval-after-epochs is overriden).                                                 |
| eval-segments-dir           | String | Path to x-vector timing info directory.                                                                                                |
| eval-xvec-ark-dir           | String | Path to Kaldi x-vectors ark files directory.                                                                                           |
| exp-name-tag                | String | Tag that appears at the end of the tensorboard log filename.                                                                           |
| Fa                          | Float  | Hyperparameter for VBx (check the publication).                                                                                        |
| Fb                          | Float  | Hyperparameter for VBx (check the publication).                                                                                        |
| gt-label-type               | String | Type of ground truth labels: PROBABILITIES, ZERO_ONES, SINGLE_SPK.                                                                     |
| in-eval-gt-rttm-dir         | String | Path to eval set ground truth rttm directory.                                                                                          |
| in-eval-GTlabels-dir        | String | Path to eval set ground truth labels directory.                                                                                        |
| in-eval-INITlabels-dir      | String | Path to eval AHC initial labels directory.                                                                                             |
| in-eval-list                | String | Path to a list containing evaluation file names.                                                                                       |
| in-GTlabels-dir             | String | Path to train ground truth labels directory.                                                                                           |
| in-INITlabels-dir           | String | Path to train initialization (currently AHC) labels directory.                                                                         |
| in-train-gt-rttm-dir        | String | Path to train set ground truth rttm directory.                                                                                         |
| in-trainlist                | String | Path to text file containing training file names.                                                                                      |
| in-val-gt-rttm-dir          | String | Path to validation set ground truth rttm directory.                                                                                    |
| in-val-GTlabels-dir         | String | Path to validation ground truth labels directory.                                                                                      |
| in-vallist                  | String | Path to text file containing validation file names.                                                                                    |
| init-model-path             | String | Path to initial model checkpoint.                                                                                                      |
| init-smoothing              | Float  | AHC label smoothing default value.                                                                                                     |
| lda-dim                     |  Int   | Number of LDA dimensions the x-vectors for VBx are reduced to.                                                                         |
| loopP                       | Float  | Hyperparameter for VBx (check the publication).                                                                                        |
| num-threads-per-worker      |  Int   | Number of threads per a single worker (PyTorch op parallelization).                                                                    |
| plda-file                   | String | Path to the Kaldi PLDA model file.                                                                                                     |
| plot-gammas                 |  Bool  | If present, system will plot gammas throughout the training to tensorbord.                                                             |
| run-dist                    |  Bool  | If torchrun is used, this flag needs to be present.                                                                                    |
| save-checkpoint-after-steps |  Int   | Number of training steps after which model checkpoint is saved (default is 1 epoch).                                                   |
| run-dist                    |  Bool  | If present, the training is distributed (parallel) and must be run using torchrun. Otherwise, the training process is single-threaded. |
| segments-dir                | String | Path to x-vector timing info directory.                                                                                                |
| tb-path                     | String | Path to tensorboard log.                                                                                                               |
| threshold                   | Float  | AHC bias.                                                                                                                              |
| use-gmm                     |  Bool  | If present, system will use GMM instead of HMM no matter the loopP value.                                                              |
| val-segments-dir            | String | Path to validation x-vector timing info directory.                                                                                     |
| val-xvec-ark-dir            | String | Path to validation Kaldi x-vectors ark files directory.                                                                                |
| xvec-ark-dir                | String | Path to Kaldi x-vectors ark files directory.                                                                                           |
| xvec-transform              | String | Path to x-vector transformation h5 file.                                                                                               |

### Inference

To run the inference with an already trained model, adjust the following flag values and paths (their meaning is the same as for the training script). 
If `--in-INITlabels-dir` is not present, system will first run AHC to obtain initial labels and then continue with VBx.

```shell
torchrun --nnodes 1
         --nproc_per_node 8
         vbhmm_infer.py
         --in-file-list data_lists/callhone_part2.txt
         --in-INITlabels-dir experiment_dir/out_dir_AHC/
         --lda-dim 128
         --model-path path_to_model_checkpoint
         --out-rttm-dir experiment_dir
         --plda-file VBx/models/ResNet101_8kHz/plda
         --segments-dir xvector_dir/segments/
         --threshold -0.015
         --use-gmm
         --xvec-ark-dir xvector_dir/xvectors/
         --xvec-transform VBx/models/ResNet101_8kHz/transform.h5
```
### Citations
In case of using the software, please cite: [Discriminative Training of VBx Diarization](https://arxiv.org/abs/2310.02732)

```
@misc{klement2023discriminative,
      title={Discriminative Training of VBx Diarization}, 
      author={Dominik Klement and Mireia Diez and Federico Landini and Lukáš Burget and Anna Silnova and Marc Delcroix and Naohiro Tawara},
      year={2023},
      eprint={2310.02732},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
### Licence
This software is licensed under the MIT licence (see `LICENSE` file).

### Contact
If you have any comments or questions, please contact: [xkleme15@vutbr.cz](mailto:xkleme15@vutbr.cz), [landini@fit.vutbr.cz](mailto:landini@fit.vutbr.cz) or [mireia@fit.vutbr.cz](mailto:mireia@fit.vutbr.cz).