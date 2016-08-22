#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/faster_rcnn_end2end_houzzdata1.sh 0 VGG16 --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"
# ./experiments/scripts/faster_rcnn_end2end_resnet_houzzdata1_tte.sh 1 RESNET --set TRAIN.SNAPSHOT_INFIX n0047 TRAIN.SCALES "[400,500,600]"
# ./experiments/scripts/faster_rcnn_end2end_resnet_houzzdata1_tte.sh 1 RESNET --set TRAIN.SNAPSHOT_INFIX n0047 TRAIN.FG_THRESH 0.8
# ./experiments/scripts/faster_rcnn_end2end_resnet_houzzdata1_tte.sh 1 RESNET --set TRAIN.SNAPSHOT_INFIX n0064 TRAIN.SCALES "[400,600,800,1000]" TRAIN.MAX_SIZE 1200
# ./experiments/scripts/faster_rcnn_end2end_resnet_houzzdata1_tte.sh 0 RESNET --set TRAIN.SNAPSHOT_INFIX n0082


set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
#ITERS=70000
#ITERS=136000
#ITERS=90000
#ITERS=100000
ITERS=150000
#ITERS=196000
#ITERS=80

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
NET_INIT=data/resnet_models/ResNet-50-model.caffemodel
#NET_INIT=data/resnet_models/ResNet-101-model.caffemodel
#NET_INIT=output/faster_rcnn_end2end/houzzdata1_train/vgg16_faster_rcnn_t0000_iter_136000.caffemodel
#NET_INIT=output/faster_rcnn_end2end/houzzdata1_train/vgg16_faster_rcnn_n0001_iter_120000.caffemodel
#NET_INIT=output/faster_rcnn_end2end/houzzdata1_train/vgg16_faster_rcnn_n0016_iter_90200.caffemodel
#NET_INIT=output/faster_rcnn_end2end/houzzdata1_train/resnet50_faster_rcnn_n0044_iter_100000.caffemodel
#NET_INIT=output/faster_rcnn_end2end/houzzdata1_train/resnet50_faster_rcnn_n0063_iter_120000.caffemodel
#NET_INIT=output/faster_rcnn_end2end/houzzdata1_train/resnet50_faster_rcnn_v8_n0080_iter_150000.caffemodel

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/houzzdata/${NET}/faster_rcnn_end2end_houzzdata1/solver50_v8_gen.prototxt \
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

#DEST_PATH_ROOT=work_data/faster-rcnn/RESNET/dets_end2end_n0044_train_nosquare_i100000_houzzdata1_val_nosquare
#DEST_PATH_ROOT=work_data/faster-rcnn/RESNET/dets_end2end_n0062_train_nosquare_i150000_houzzdata1_val_nosquare
#DEST_PATH_ROOT=work_data/faster-rcnn/RESNET/dets_end2end_n0064_train_nosquare_i150000_houzzdata1_val_nosquare
DEST_PATH_ROOT=work_data/faster-rcnn/RESNET/dets_end2end_n0082_train_nosquare_i${ITERS}_houzzdata1_val_nosquare
SCALE_NUM=1
CONF_THRESH=0.01
PROTOTXT=models/houzzdata/RESNET/faster_rcnn_end2end_houzzdata1/test50_v8_gen_frz-conv3.prototxt
#PROTOTXT=models/houzzdata/RESNET/faster_rcnn_end2end_houzzdata1/test101.prototxt
IMG_LIST=data/HouzzDataCollection/HouzzData1/ImageLists/val.txt
#IMG_LIST=data/HouzzDataCollection/HouzzData1/ImageLists/val3.txt
IMG_PATH=data/HouzzDataCollection/HouzzData1/Images/
RECORD_NAME=detection_record.txt
IMG_EXT=.jpg

time ./tools/batch_detect.py \
    --gpu ${GPU_ID} --model ${CAFFEMODEL} --conf ${CONF_THRESH} \
    --scl ${SCALE_NUM} --proto ${PROTOTXT} --list ${IMG_LIST} \
    --imgpath ${IMG_PATH} --rec ${RECORD_NAME} --destrt ${DEST_PATH_ROOT} \
    --ext ${IMG_EXT}

#SCALE_NUM=4
#CONF_THRESH=0.5
##PROTOTXT=models/houzzdata/VGG16/faster_rcnn_end2end_houzzdata1/test.prototxt
#PROTOTXT=models/houzzdata/RESNET/faster_rcnn_end2end_houzzdata1/test50.prototxt
##PROTOTXT=models/houzzdata/RESNET/faster_rcnn_end2end_houzzdata1/test101.prototxt
#IMG_LIST=data/HouzzDataCollection/HouzzData1/ImageLists/val.txt
##IMG_LIST=data/HouzzDataCollection/HouzzData1/ImageLists/val3.txt
#IMG_PATH=data/HouzzDataCollection/HouzzData1/Images/
#RECORD_NAME=detection_record.txt
#IMG_EXT=.jpg

#time ./tools/batch_detect.py \
    #--gpu ${GPU_ID} --model ${CAFFEMODEL} --conf ${CONF_THRESH} \
    #--scl ${SCALE_NUM} --proto ${PROTOTXT} --list ${IMG_LIST} \
    #--imgpath ${IMG_PATH} --rec ${RECORD_NAME} --destrt ${DEST_PATH_ROOT} \
    #--ext ${IMG_EXT}
