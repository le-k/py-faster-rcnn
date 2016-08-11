#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/faster_rcnn_end2end_mini.sh 0 ZF \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"
# ./experiments/scripts/faster_rcnn_end2end_mini.sh 0 VGG16 \
#   --set SNAPTSHOT_INFIX t0001

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
#ITERS=70000
ITERS=200
#DATASET_TRAIN=voc_2007_trainval
#DATASET_TEST=voc_2007_test
DATASET_TRAIN=houzzdata0_mini
#DATASET_TEST=houzzdata0_val

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo "variable array=$array"
echo "variable len=$len"
echo "variable EXTRA_ARGS=$EXTRA_ARGS"
echo "variable EXTRA_ARGS_SLUG=$EXTRA_ARGS_SLUG"

LOG="experiments/logs/faster_rcnn_mini_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#NET_INIT=data/imagenet_models/${NET}.v2.caffemodel
#NET_INIT=output/faster_rcnn_end2end/houzzdata0_train/vgg16_faster_rcnn_iter_110000_train_h1_34000.caffemodel
NET_INIT=output/faster_rcnn_alt_opt/houzzdata1_train/vgg16_fast_rcnn_stage2_iter_68000_houzzdata1_train_34000.caffemodel

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

#time ./tools/test_net.py --gpu ${GPU_ID} \
  #--def models/${NET}/faster_rcnn_end2end/test.prototxt \
  #--net ${NET_FINAL} \
  #--imdb ${DATASET_TEST} \
  #--cfg experiments/cfgs/faster_rcnn_end2end.yml \
  #${EXTRA_ARGS}
