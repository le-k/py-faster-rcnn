#!/bin/bash
# Usage:
# Example:
#python tools/myDemo8.py --gpu 0  --conf 0.1 --list tools/all_space_photos_03102016_filelist_aa 2>&1 | tee rst_all_space_photos_03102016_ab.txt

#set -x
#set -e

export PYTHONUNBUFFERED="True"

#time ./tools/myDemo8.py --gpu 0 --conf 0.2 --end 96000 --list tools/tag_type2_spaces 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/type2_parts/rst_type2_filelist_1-96000.txt

#time ./tools/myDemo8.py --gpu 1 --conf 0.2 --start 96001 --list tools/tag_type2_spaces 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/type2_parts/rst_type2_filelist_96001-191852.txt

time ./tools/myDemo8.py --gpu 1 --conf 0.2 --start 96000 --end 96001 --list tools/tag_type2_spaces 2>&1 | tee work_data/faster-rcnn/VGG16/all_photos/type2_parts/rst_type2_filelist_96001-96002.txt


