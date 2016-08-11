#!/bin/bash

TASK=$1
MACHINE=$2
CREDENTIAL=""
#ROOT_PARENT=""
#For backing up large data files, we use sync with caution, because when data files change, their sizes change, too. For backing up scripts, sync can be dangerous.
if [ "$MACHINE" == "deep" ] #back up data
then
    #ROOT_PARENT="/root/packages"
    CREDENTIAL="--profile=recognition"
elif [ "$MACHINE" == "aws" ]
then
    #ROOT_PARENT="/home/ubuntu"
    CREDENTIAL="--profile=default"
else
    echo "Unrecognized machine type. Please use deep or aws"
    exit 1
fi

echo "Using credential [ $CREDENTIAL ]"


CWD_BASE=${PWD##*/}
FRCNN_ROOT="py-faster-rcnn"
if [ "$CWD_BASE" != "$FRCNN_ROOT" ]
then
    echo "Current directory wrong. Please navigate to $FRCNN_ROOT. "
    exit 1
fi


#For backing up large data files, we use sync with caution, because when data files change, their sizes change, too. For backing up scripts, sync can be dangerous.
if [ "$TASK" == "d" ] #back up data
then
    aws s3 sync ./data/HouzzDataCollection/ s3://houzz-archives-development/recognition/data/faster-rcnn-backup/data/HouzzDataCollection/  --exclude "HouzzData0*" --exclude "*.jpg" --exclude "*/Images/*" --exclude "*/Annotations/*" --exclude "*/ImageLists/*" --exclude "*.tar.gz"  "$CREDENTIAL"  #seems to examine too many files recursively, without short circuit
    #aws s3 sync ./data/HouzzDataCollection/ s3://houzz-archives-development/recognition/data/faster-rcnn-backup/data/HouzzDataCollection/ --exclude "*" --include "*.tar.gz" --dryrun --profile=recognition #seems to examine too many files recursively, without short circuit
    #aws s3 cp ./data/HouzzDataCollection/HouzzData1.tar.gz s3://houzz-archives-development/recognition/data/faster-rcnn-backup/data/HouzzDataCollection/  "$CREDENTIAL"

elif [ "$TASK" == "w" ] #back up work_data
then
    aws s3 sync ./work_data/ s3://houzz-archives-development/recognition/data/faster-rcnn-backup/work_data/ --exclude ".git*"  "$CREDENTIAL"

elif [ "$TASK" == "m" ] #back up models
then
    aws s3 sync ./output/ s3://houzz-archives-development/recognition/data/faster-rcnn-backup/models/ --exclude ".git*"  "$CREDENTIAL"

elif [ "$TASK" == "l" ] #back up logs
then
    aws s3 sync ./experiments/logs/ s3://houzz-archives-development/recognition/data/faster-rcnn-backup/logs/ "$CREDENTIAL"

#elif [ "$TASK" == "s" ] #backup scripts
#then
    ##copy scripts (with max size limit) from working directory to repo
    #rsync -rav --delete ./ /home/lekang/houzz/research/vision/detection/py-faster-rcnn/ --exclude='/caffe-fast-rcnn/' --exclude='/data/' --exclude='/work_data'  --exclude='/output/' --exclude='/tmp*' --exclude='/experiments/logs/' --include='/lib/datasets/' --include='/lib/fast_rcnn/' --exclude='/lib/*' --exclude='*.pkl' --exclude='.git*' --exclude='*.tgz' --exclude='*.swp'  --exclude='*.pyc'  --max-size=1m
    ##alternative way
    ##rsync -rav ./ /home/lekang/houzz/research/vision/detection/py-faster-rcnn/  --exclude='*.pkl' --exclude='.git*' --exclude='*.tgz' --exclude='*.swp'  --exclude='*.pyc' --include='/lib/' --include='/lib/datasets/***' --include='/lib/fast_rcnn/***' --include='/tools/***' --include='/models/' --include='/experiments/' --include='experiments/scripts/***' --include='/experiments/cfgs/***'  --exclude='*' --max-size=1m

else
    echo "Not a valid argument -- use 'l' for logs and 'd' for data"
fi
