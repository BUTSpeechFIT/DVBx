#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p exp

INIT_DIR=/mnt/matylda6/mireia/NTT/DiscriminativeVBx/Experiments/Callhome/Initialization2/
LIST=/mnt/matylda4/landini/data/callhome1/lists/all.txt
#LIST=/mnt/matylda6/mireia/GithubRep/VBx_pytorch/VBx/all_CALLHOME.txt
LAB_DIR=/mnt/matylda4/landini/data/CALLHOME/labs/VAD/  # lab files directory with VAD segments
#LIST=/mnt/matylda6/mireia/GithubRep/VBx_pytorch/VBx/iaaa.txt
REF_DIR=/mnt/matylda4/landini/data/CALLHOME/rttms/ref/

#for audio in $(ls example/audios/16k)
#do
      #filename=$(echo "${audio}" | cut -f 1 -d '.')
      filename=iaaa

      #echo ${filename} > exp/list.txt

      # run variational bayes on top of x-vectors
      python VBx/vbhmm_train.py \
          --out-rttm-dir exp \
          --in-trainlist $LIST \
          --in-GTlabels-dir $INIT_DIR/xvector_GT_labels/ \
          --in-AHClabels-dir $INIT_DIR/out_dir_AHC/ \
          --xvec-ark-dir $INIT_DIR/xvecs/xvectors/ \
          --segments-dir $INIT_DIR/xvecs/segments/ \
          --xvec-transform VBx/models/ResNet101_8kHz/transform.h5 \
          --plda-file VBx/models/ResNet101_8kHz/plda \
          --threshold -0.015 \
          --lda-dim 128 \
          --Fa 0.3 \
          --Fb 17.0 \
          --loopP 0.4

      # check if there is ground truth .rttm file
      if [ -f example/rttm/${filename}.rttm ]
      then
          # run dscore
          python dscore/score.py -r example/rttm/${filename}.rttm -s exp/${filename}.rttm --collar 0.25 --ignore_overlaps
      fi
#done
