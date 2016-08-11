#!/usr/bin/env python
import _init_paths
import os
import numpy as np
import cPickle as pkl
from IPython import embed
import itertools
from utils.timer import Timer
import argparse
import subprocess
import sys




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--pdir', dest='predict_dir', help='Directory of prediction results')
    parser.add_argument('--gtype', dest='gt_type', help='Ground truth type to use [1]',
                        default=1, type=int)
    parser.add_argument('--gdir', dest='gtruth_dir', help='Ground truth directory')
    parser.add_argument('--conf', dest='conf_thresh', help='Confidence threshold to use [0.2]',
                        default = 0.2, type=float)
    parser.add_argument('--list', dest='test_img_list', help='List of test images')
    parser.add_argument('--idir', dest='img_dir', help='Image direcotry')
    parser.add_argument('--c2c', dest='cate2cls_file', help='File name for mapping category to class index')
    parser.add_argument('--rep', dest='report_file', help='Report file')
    parser.add_argument('--sbs', dest='sbs', help='Whether to generate side by side comparison images',
                        default=0, type=int)

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()
    #predict_dir = "./work_data/faster-rcnn/VGG16/dets_end2end_n0017_train_nosquare_i50000_houzzdata1_val_nosquare/t0.2"
    #predict_dir = "./work_data/faster-rcnn/VGG16/dets_end2end_n0030_train_nosquare_i80000_houzzdata1_val_nosquare/t0.2"
    #predict_dir = "./work_data/faster-rcnn/RESNET/dets_end2end_n0021_train_nosquare_i60000_houzzdata1_val_nosquare/t0.2"
    #predict_dir = "./work_data/faster-rcnn/VGG16/dets_end2end_n0018_train_nosquare_i90000_houzzdata1_val_nosquare_mulscl1/t0.2"
    #predict_dir = "./work_data/faster-rcnn/RESNET/dets_end2end_n0046_train_nosquare_i200000_houzzdata1_val_nosquare/mulscl1/t0.2"
    predict_dir = "./work_data/faster-rcnn/RESNET/dets_end2end_n0047_train_nosquare_i200000_houzzdata1_val_nosquare/mulscl3/t0.5"

    #conf_thresh = 0.2
    conf_thresh = 0.5
    #conf_thresh = 0.1
    #conf_thresh = 0.001

    #gt_type = 0 #x and y are left upper corner
    #gtruth_dir = "./data/HouzzDataCollection/HouzzData0/Annotations"
    #test_img_list = "./data/HouzzDataCollection/HouzzData0/ImageLists/val.txt"
    #img_dir = "./data/HouzzDataCollection/HouzzData0/Images"
    #cate2cls_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_table_purged.txt"

    gt_type = 1 #xc and yc are center
    #gtruth_dir = "./data/HouzzDataCollection/HouzzData1/Annotations"
    gtruth_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged_nosquare"
    #test_img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/val.txt"
    test_img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/val_nosquare.txt"
    #test_img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/test.txt"
    #test_img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/test_nosquare.txt"
    img_dir = "./data/HouzzDataCollection/HouzzData1/Images"
    cate2cls_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/cat_table_purged.txt"

    #report_file = predict_dir + "_eval_report.txt" if not ignore_cate else \
                    #predict_dir + "_eval_report_ignore_cate.txt"
    #report_file = predict_dir + "_eval_report_all.txt" 
    #report_file = predict_dir + "_eval_report_th{:.2f}_detail.txt".format(conf_thresh)
    report_file = predict_dir + "_eval_report_th{:.2f}_nosquare.txt".format(conf_thresh)
    sbs = 0
    
    input_args =  "--pdir {predict_dir} --gtype {gt_type} --gdir {gtruth_dir} \
            --conf {conf_thresh} --list {test_img_list} --idir {img_dir} \
            --c2c {cate2cls_file} --rep {report_file} --sbs {sbs}".format(\
            predict_dir=predict_dir, gt_type=gt_type, gtruth_dir=gtruth_dir, \
            conf_thresh=conf_thresh, test_img_list=test_img_list, img_dir=img_dir, \
            cate2cls_file=cate2cls_file, report_file=report_file, sbs=sbs)
    
    try:
        retcode = subprocess.call("python tools/eval_dets.py " + input_args, shell=True)
        if retcode < 0:
            print >> sys.stderr, "Eval was terminated by signal", -retcode
        else:
            print >> sys.stderr, "Eval returned ", retcode
    except OSError as e:
        print >> sys.stderr, "Eval failed:", e

