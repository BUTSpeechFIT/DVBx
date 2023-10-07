#!/bin/bash

INSTRUCTION=$1

exp_dir=$2 # output experiment directory
xvec_dir=$3 # output xvectors directory

WAV_DIR= # wav files directory
FILE_LIST= # txt list of files to process
LAB_DIR= # lab files directory with VAD segments
REF_DIR= # reference rttm files directory

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


if [[ $INSTRUCTION = "xvectors" ]]; then
	WEIGHTS_DIR=$DIR/VBx/models/ResNet101_16kHz/nnet
	if [ ! -f $WEIGHTS_DIR/raw_81.pth ]; then
	    cat $WEIGHTS_DIR/raw_81.pth.zip.part* > $WEIGHTS_DIR/unsplit_raw_81.pth.zip
		unzip $WEIGHTS_DIR/unsplit_raw_81.pth.zip -d $WEIGHTS_DIR/
	fi

	WEIGHTS=$DIR/VBx/models/ResNet101_16kHz/nnet/raw_81.pth
	EXTRACT_SCRIPT=$DIR/VBx/extract.sh
	DEVICE=cpu

	mkdir -p $xvec_dir/xvector_GT_labels
	$EXTRACT_SCRIPT ResNet101 $WEIGHTS $WAV_DIR $LAB_DIR $FILE_LIST $xvec_dir $DEVICE $REF_DIR $xvec_dir/xvector_GT_labels
         
	# Replace this to submit jobs to a grid engine
	bash $xvec_dir/xv_task
fi


BACKEND_DIR=$DIR/VBx/models/ResNet101_16kHz
if [[ $INSTRUCTION = "diarization" ]]; then
	TASKFILE=$exp_dir/diar_AHC_task
	OUTFILE=$exp_dir/diar_AHC_out
	rm -f $TASKFILE $OUTFILE
	mkdir -p $exp_dir/lists

	thr=-0.015
	lda_dim=128
	OUT_DIR=$exp_dir/out_dir_AHC
	if [[ ! -d $OUT_DIR ]]; then
		mkdir -p $OUT_DIR
        else
                echo "OUT_DIR existed, overwriting files"
        fi
        while IFS= read -r line; do
		grep $line $FILE_LIST > $exp_dir/lists/$line".txt"
		echo "python $DIR/initAHC.py --out-AHCinit-dir $OUT_DIR --xvec-ark-file $xvec_dir/xvectors/$line.ark --segments-file $xvec_dir/segments/$line --plda-file $BACKEND_DIR/plda --xvec-transform $BACKEND_DIR/transform.h5 --threshold $thr" >> $TASKFILE
	done < $FILE_LIST
	bash $TASKFILE > $OUTFILE

fi
