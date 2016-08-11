#!/bin/bash
#To mount the directories in the repo to the working directory

MACHINE=$1
ROOT_PARENT=""
#For backing up large data files, we use sync with caution, because when data files change, their sizes change, too. For backing up scripts, sync can be dangerous.
if [ "$MACHINE" == "deep" ] #back up data
then
    ROOT_PARENT="/root/packages"
elif [ "$MACHINE" == "aws" ]
then
    ROOT_PARENT="/home/ubuntu"
else
    echo "Unrecognized machine type. Please use deep or aws"
    exit 1
fi

#Check current directory
CWD_BASE=${PWD##*/}
FRCNN_ROOT="py-faster-rcnn"
if [ "$CWD_BASE" != "$FRCNN_ROOT" ]
then
    echo "Current directory wrong. Please navigate to the actual working directory $FRCNN_ROOT. "
    exit 1
fi

#Prerequisite -- remove these actual directories/symlinks: ./models, ./tool, ./experiments/scripts, ./experiments/cfgs, ./lib/datasets, ./lib/fast_rcnn
#check if dir already exists
echo "Checking directories..."
DIRS=("models" "tools" "experiments/cfgs" "experiments/scripts" "lib/datasets" "lib/fast_rcnn")
for t in "${DIRS[@]}"; do
    if [ -d "./$t" ]; then
        echo "Directory $t already exists! Stopped."
        exit
    else
        echo "./$t is not there. Good."
    fi
done
echo "Creating directorie..."
for t in "${DIRS[@]}"; do
    mkdir -p "./$t"
    echo "  ./$t"
done

echo "Mount bind directories..."
for t in "${DIRS[@]}"; do
    echo "$t"
    sudo mount --bind /home/lekang/houzz/research/vision/detection/py-faster-rcnn/"$t" ./"$t"
done

echo "Adding lines to /etc/fstab..."
#sudo echo "/home/lekang/houzz/research/vision/detection/py-faster-rcnn/models /root/packages/py-faster-rcnn/models none rw,bind 0 0
#/home/lekang/houzz/research/vision/detection/py-faster-rcnn/tools /root/packages/py-faster-rcnn/tools none rw,bind 0 0
#/home/lekang/houzz/research/vision/detection/py-faster-rcnn/experiments/scripts /root/packages/py-faster-rcnn/experiments/scripts none rw,bind 0 0
#/home/lekang/houzz/research/vision/detection/py-faster-rcnn/experiments/cfgs /root/packages/py-faster-rcnn/experiments/cfgs none rw,bind 0 0
#/home/lekang/houzz/research/vision/detection/py-faster-rcnn/lib/datasets /root/packages/py-faster-rcnn/lib/datasets none rw,bind 0 0
#/home/lekang/houzz/research/vision/detection/py-faster-rcnn/lib/fast_rcnn /root/packages/py-faster-rcnn/lib/fast_rcnn none rw,bind 0 0" | sudo tee --append /etc/fstab

for t in "${DIRS[@]}"; do
    sudo echo "/home/lekang/houzz/research/vision/detection/py-faster-rcnn/$t /root/packages/py-faster-rcnn/$t none rw,bind 0 0\n" | sudo tee --append /etc/fstab
done
