#!/bin/bash
# Usage:
# Example:
# ./tools/run_myDemo8fly.sh 1

#set -x
#set -e

export PYTHONUNBUFFERED="True"


echo "$1"

if [ "$1" == '1' ]
then 
    time ./tools/myDemo8fly.py --gpu 0 --conf 0.2 --start 99466 --end 462883 --list tools/all_space_photos_07152016_notag4_urls 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/all_07152016/rst_all_space_photos_07152016_notag4_urls_99466-462883.txt
elif [ "$1" == '2' ]
then
    time ./tools/myDemo8fly.py --gpu 0 --conf 0.2 --start 462883 --end 826300 --list tools/all_space_photos_07152016_notag4_urls 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/all_07152016/rst_all_space_photos_07152016_notag4_urls_462883-826300.txt
elif [ "$1" == '3' ]
then
    time ./tools/myDemo8fly.py --gpu 0 --conf 0.2 --start 826300 --end 1189717 --list tools/all_space_photos_07152016_notag4_urls 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/all_07152016/rst_all_space_photos_07152016_notag4_urls_826300-1189717.txt
elif [ "$1" == '4' ]
then
    time ./tools/myDemo8fly.py --gpu 0 --conf 0.2 --start 1189717 --end 1553134 --list tools/all_space_photos_07152016_notag4_urls 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/all_07152016/rst_all_space_photos_07152016_notag4_urls_1189717-1553200.txt
else
    echo "wrong session number"
fi

#time ./tools/myDemo8fly.py --gpu 1 --conf 0.2 --start 1553200 --end 1649798 --list tools/all_space_photos_07152016_notag4_urls 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/all_07152016/rst_all_space_photos_07152016_notag4_urls_1553200-1649798.txt
#time ./tools/myDemo8fly.py --gpu 1 --conf 0.2 --start 1649798 --end 2013915 --list tools/all_space_photos_07152016_notag4_urls 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/all_07152016/rst_all_space_photos_07152016_notag4_urls_1649798-2013915.txt
#time ./tools/myDemo8fly.py --gpu 1 --conf 0.2 --start 2013915 --end 2378032 --list tools/all_space_photos_07152016_notag4_urls 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/all_07152016/rst_all_space_photos_07152016_notag4_urls_2013915-2378032.txt
#time ./tools/myDemo8fly.py --gpu 1 --conf 0.2 --start 2378032 --end 2742149 --list tools/all_space_photos_07152016_notag4_urls 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/all_07152016/rst_all_space_photos_07152016_notag4_urls_2378032-2742149.txt
#time ./tools/myDemo8fly.py --gpu 1 --conf 0.2 --start 2742149 --list tools/all_space_photos_07152016_notag4_urls 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/all_07152016/rst_all_space_photos_07152016_notag4_urls_2742149-3106268.txt

