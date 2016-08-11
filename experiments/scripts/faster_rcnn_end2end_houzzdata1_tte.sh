#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/faster_rcnn_end2end_houzzdata1.sh 0 VGG16 \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"
# ./experiments/scripts/faster_rcnn_end2end_houzzdata1_tte.sh 0 VGG16  --set TRAIN.SNAPSHOT_INFIX n0033
# ./experiments/scripts/faster_rcnn_end2end_houzzdata1_tte.sh 0 VGG16  --set TRAIN.SNAPSHOT_INFIX n0033 TRAIN.SCALES "[400,600,800]"
# ./experiments/scripts/faster_rcnn_end2end_houzzdata1_tte.sh 0 VGG16  --set TRAIN.SNAPSHOT_INFIX n0033 TRAIN.SCALES "[400,600,800]" TRAIN.MAX_SIZE 1400
# ./experiments/scripts/faster_rcnn_end2end_houzzdata1_tte.sh 0 VGG16  --set TRAIN.SNAPSHOT_INFIX n0033 TRAIN.FG_THRESH 0.8

# ./experiments/scripts/faster_rcnn_end2end_houzzdata1_tte.sh 0 VGG16  --set TRAIN.SNAPSHOT_INFIX n0099

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
#ITERS=70000
#ITERS=136000
#ITERS=80000
#ITERS=100000
#ITERS=150000
ITERS=30
#ITERS=196000

#DATASET_TRAIN=voc_2007_trainval
#DATASET_TEST=voc_2007_test
#DATASET_TRAIN=houzzdata0_train
DATASET_TRAIN=houzzdata1_train
#DATASET_TRAIN=houzzdata1_mini
#DATASET_TEST=houzzdata0_val

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/faster_rcnn_end2end_houzzdata1_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#NET_INIT=data/imagenet_models/${NET}.v2.caffemodel
#NET_INIT=data/faster_rcnn_models/${NET}_faster_rcnn_final.caffemodel
#NET_INIT=output/faster_rcnn_alt_opt/houzzdata1_train/vgg16_fast_rcnn_stage2_iter_68000_houzzdata1_train_34000.caffemodel
#NET_INIT=output/faster_rcnn_end2end/houzzdata1_train/vgg16_faster_rcnn_t0000_iter_136000.caffemodel
#NET_INIT=output/faster_rcnn_end2end/houzzdata1_train/vgg16_faster_rcnn_n0001_iter_120000.caffemodel
NET_INIT=output/faster_rcnn_end2end/houzzdata1_train/vgg16_faster_rcnn_n0016_iter_90200.caffemodel

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/houzzdata/${NET}/faster_rcnn_end2end_houzzdata1/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

set +x
#NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
CAFFEMODEL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

#time ./tools/test_net.py --gpu ${GPU_ID} \
  #--def models/${NET}/faster_rcnn_end2end/test.prototxt \
  #--net ${NET_FINAL} \
  #--imdb ${DATASET_TEST} \
  #--cfg experiments/cfgs/faster_rcnn_end2end.yml \
  #${EXTRA_ARGS}

DEST_PATH_ROOT=work_data/faster-rcnn/VGG16/dets_end2end_n0099_train_nosquare_i30_houzzdata1_val_nosquare
SCALE_NUM=1
CONF_THRESH=0.2
PROTOTXT=models/houzzdata/VGG16/faster_rcnn_end2end_houzzdata1/test.prototxt
IMG_LIST=data/HouzzDataCollection/HouzzData1/ImageLists/val.txt
IMG_PATH=data/HouzzDataCollection/HouzzData1/Images/
RECORD_NAME=detection_record.txt
IMG_EXT=.jpg

time ./tools/batch_detect.py \
    --gpu ${GPU_ID} --model ${CAFFEMODEL} --conf ${CONF_THRESH} \
    --scl ${SCALE_NUM} --proto ${PROTOTXT} --list ${IMG_LIST} \
    --imgpath ${IMG_PATH} --rec ${RECORD_NAME} --destrt ${DEST_PATH_ROOT} \
    --ext ${IMG_EXT}

