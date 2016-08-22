#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import time
from IPython import embed

class global_meta:
    def __init__(self):
        self.no_detection_img_list = []
        self.no_detection_img_ct = 0
        self.detection_ct = 0
        self.conf_thresh = 0.0
        #self.banned_categories = set( ['tabletop'] )
        self.banned_categories = set()
        self.nms_thresh_within_class = 0.3
        self.nms_thresh_all = 0.4
        self.target_size = 600
        self.max_size = 1000


GM = global_meta()


def get_classes():
    #cate_table_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_table_merged_modified.txt"
    cate_table_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/cat_table_purged.txt"
    all_classes = ['__background__']
    with open(cate_table_file, 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            all_classes.append(vals[-1])
    #print all_classes
    return all_classes

    
#CLASSES = ('__background__',
           #'aeroplane', 'bicycle', 'bird', 'boat',
           #'bottle', 'bus', 'car', 'cat', 'chair',
           #'cow', 'diningtable', 'dog', 'horse',
           #'motorbike', 'person', 'pottedplant',
           #'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = get_classes()

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


    
def write_detections(boxfile_pathname, class_inds, dets):
    """Write detected bounding boxes to txt files."""
    inds = range(dets.shape[0])
    ct = 0
    with open(boxfile_pathname, 'w') as fh:
        for i in inds:
            ct += 1
            bbox = dets[i, :4]
            score = dets[i, -1]
            x = int(bbox[0])
            y = int(bbox[1])
            wid = int(bbox[2] - bbox[0] + 1)
            hei = int(bbox[3] - bbox[1] + 1)
            fh.write("{:d} {:d} {:d} {:d} {:d} {:0.4f}\n".format(
                x, y, wid, hei, class_inds[i], score))




def vis_detections(dest_path, image_basename, im, class_inds, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = range(dets.shape[0])
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    ct = 0
    for i in inds:
        if class_inds[i] in GM.banned_categories:
            continue
        ct += 1
        bbox = dets[i, :4]
        score = dets[i, -1]
        x = int(bbox[0])
        y = int(bbox[1])
        wid = int(bbox[2] - bbox[0])
        hei = int(bbox[3] - bbox[1])
        #fh.write("{:d} {:d} {:d} {:d} {:d}\n".format(
            #x, y, wid, hei, class_inds[i]))
        ax.add_patch(
            plt.Rectangle( (x, y), wid, hei, fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:d}. {:s} {:.3f}'.format(ct, CLASSES[class_inds[i]], score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        GM.detection_ct += 1

    ax.set_title(('{} detections with '
                 'confidence >= {:.2f}').format(ct, thresh), fontsize=14)
    print('{} detections'.format(ct))

    plt.axis('off')
    plt.tight_layout()
    #plt.draw()
    plt.savefig(os.path.join(dest_path, image_basename ))
    plt.close()




def scale_image(im, target_size, max_size):
    """scale an image for detection"""
    im = im.astype(np.float32, copy=False)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale



def demo(net, image_name, dest_path):
    """Detect object classes in an image using pre-computed object proposals."""
    #target_size = GM.target_size
    #max_size = GM.max_size

    # Load the demo image
    #im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
    im = cv2.imread(image_name)
    #im_hei, im_wid = im.shape[:2] 
    #im, scale = scale_image(im, target_size, max_size)
    # Detect all object classes and regress object bounds
    timer1 = Timer() #detection tot time (core + nms)
    timer2 = Timer() #detection core time
    timer1.tic()
    scales_tot = len(GM.test_scale_list)
    mul_dets_list = [None] * scales_tot
    mul_cls_list = [None] * scales_tot
    valid_scale_ct = 0 #count number of scales that have valid detections

    for i, target_size, max_size in zip(range(scales_tot), GM.test_scale_list, GM.max_size_list):
        timer2.tic()

        cfg.TEST.SCALES = [target_size]
        cfg.TEST.MAX_SIZE = max_size
        scores, boxes = im_detect(net, im)

        #scores_list[i], boxes_list[i] = im_detect(net, im)
        #scores_list = [None] * scales_tot
        #boxes_list = [None] * scales_tot
        #scores = np.vstack(scores_list)
        #boxes = np.vstack(boxes_list)

        timer2.toc()

        # Visualize detections for each class
        #CONF_THRESH = 0.5
        CONF_THRESH = GM.conf_thresh
        #NMS_THRESH = 0.3
        all_dets = None
        all_cls = None

        #timer.tic()
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]

            #thresholding at early stage to reduce nms compuation
            pre_filter_ind = np.where(cls_scores >= CONF_THRESH)[0]
            cls_boxes = cls_boxes[pre_filter_ind, :]
            cls_scores = cls_scores[pre_filter_ind]

            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            #keep = nms(dets, NMS_THRESH) # nms in just this class
            keep = nms(dets, GM.nms_thresh_within_class) # nms in just this class
            dets = dets[keep, :]
            #vis_detections(im, cls, dets, dest_path, thresh=CONF_THRESH)
            if dets.shape[0] <= 0:
                continue
            if all_dets is None:
                all_dets = dets
                #all_cls = [cls] * all_dets.shape[0]
                all_cls = [cls_ind] * all_dets.shape[0]
            else:
                all_dets = np.vstack( (all_dets, dets))
                #all_cls.extend( [cls] * dets.shape[0] )
                all_cls.extend( [cls_ind] * dets.shape[0] )
            #print "all_cls: ", all_cls
            #print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                        #CONF_THRESH)
        if all_dets is None:
            continue

        all_keep = nms(all_dets, GM.nms_thresh_all) #nms among all classes
        all_dets = all_dets[all_keep, :]
        all_cls = np.array(all_cls)[all_keep]

        mul_dets_list[valid_scale_ct] = all_dets
        mul_cls_list[valid_scale_ct] = all_cls
        valid_scale_ct += 1
    
    timer1.toc()
    detect_core_time = timer2.total_time
    print ('Detection core took {:.3f}s ').format(detect_core_time)
    detect_tot_time = timer1.total_time
    print ('Detection and nms took {:.3f}s').format(detect_tot_time)

    if valid_scale_ct == 0:
        print "No detections at all!"
        GM.no_detection_img_ct += 1
        GM.no_detection_img_list.append(image_name)
        return detect_core_time, detect_core_time

    mul_dets = np.vstack(mul_dets_list[ : valid_scale_ct])
    mul_cls = np.hstack(mul_cls_list[ : valid_scale_ct])
    mul_keep = nms(mul_dets, GM.nms_thresh_all)
    mul_dets = mul_dets[mul_keep, :]
    mul_cls = mul_cls[mul_keep]

    #write all boxes to disk
    image_basename = os.path.basename(image_name)
    boxfile_pathname = os.path.join(dest_path, image_basename[:-4] + '.txt')
    write_detections(boxfile_pathname,  mul_cls, mul_dets)

    #threshold not set, do not visualize
    if CONF_THRESH <= 0:
        return detect_core_time, detect_tot_time

    #visualize boxes whose scores are above the threshold
    above_thresh = np.where(mul_dets[:, -1] >= CONF_THRESH)[0]
    #embed()
    select_cls = mul_cls[above_thresh]
    select_dets = mul_dets[above_thresh, :]
    if len(above_thresh) == 0:
        print "Nothing above threshold!"
        GM.no_detection_img_ct += 1
        GM.no_detection_img_list.append(image_name)
    else:
        vis_detections(dest_path, image_basename, im, select_cls, select_dets, thresh=CONF_THRESH)

    return detect_core_time, detect_tot_time




def set_scales(scale_num):
    if scale_num == 1: 
        #mulscl1
        GM.test_scale_list = [600,]
        GM.max_size_list = [1000,]
    elif scale_num == 2:
        #mulscl2
        GM.test_scale_list = [600, 800]
        GM.max_size_list = [1000, 1200]
    elif scale_num == 3:
        #mulscl3
        #GM.test_scale_list = [400, 600, 800]
        GM.test_scale_list = [400, 500, 600]
        GM.max_size_list = [1000, 1000, 1000]
    elif scale_num == 4:
        #mulscl4
        #GM.test_scale_list = [400, 600, 800, 1000]
        GM.test_scale_list = [400, 600, 800]
        GM.max_size_list = [1000, 1000, 1200]
    elif scale_num == 5:
        #mulscl5
        GM.test_scale_list = [200, 400, 600, 800, 1000]
        GM.max_size_list = [600, 1000, 1000, 1200, 1400]

    elif scale_num == 6:
        GM.test_scale_list = [800]
        GM.max_size_list = [1200]
    elif scale_num == 7:
        GM.test_scale_list = [1000]
        GM.max_size_list = [1200]

    elif scale_num == 8:
        GM.test_scale_list = [400]
        GM.max_size_list = [1000]
    elif scale_num == 9:
        GM.test_scale_list = [500]
        GM.max_size_list = [1000]

    else:
        raise AssertionError("Invalid scale number!")



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    #parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        #choices=NETS.keys(), default='vgg16')
    parser.add_argument('--model', dest='caffemodel', help='Model for test stage')
    parser.add_argument('--conf', dest='conf_thresh', help='Confidence threshold to use [0.5]',
                        default = 0, type=float)
    parser.add_argument('--scl', dest='scale_num', help='Number of scales to use [1]',
                        default=1, type=int)
    parser.add_argument('--proto', dest='prototxt', help='Prototxt for test stage')
    parser.add_argument('--list', dest='img_list', help='List of test images')
    parser.add_argument('--imgpath', dest='img_path', help='Path to test images')
    parser.add_argument('--rec', dest='record_name', help='Detetion record name')
    parser.add_argument('--destrt', dest='dest_path_root', help='Result path')
    parser.add_argument('--ext', dest='img_ext', help='Image extension name')


    args = parser.parse_args()

    return args




if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    GM.conf_thresh = args.conf_thresh
    set_scales(args.scale_num)
    dest_path = os.path.join(args.dest_path_root, "mulscl{}/t{:0.2f}/".format(args.scale_num, args.conf_thresh) )
    caffemodel = args.caffemodel
    prototxt = args.prototxt
    img_list = args.img_list
    img_path = args.img_path
    record_name = args.record_name
    img_ext = args.img_ext 
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id


    assert dest_path is not None
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    coreTimeTotal = 0.0
    detectTimeTotal = 0.0
    timeCostTotal = 0
    with open(img_list, 'r') as fh:
        ct = 0
        for line in fh:
            #if ct > 25:
            #    break;
            t1 = time.time()
            img_basename = line.strip() + img_ext
            img_name = os.path.join(img_path, img_basename)
            print ct, "{}".format(img_name)
            detect_core_time, detect_tot_time = demo(net, img_name, dest_path)
            coreTimeTotal += detect_core_time
            detectTimeTotal += detect_tot_time
            t2 = time.time()
            print "time cost for this image: {:.2f}s".format(t2 - t1)
            print "{} images without detections!".format(GM.no_detection_img_ct)
            timeCostTotal += t2 - t1
            ct += 1
            print "Average core time: {:.3f}".format( coreTimeTotal / ct)
            print "Average detection time: {:.3f}".format( detectTimeTotal / ct)
            print "\n"
    print "average time cost: {:.2f}s".format(float(timeCostTotal) / ct)
    #plt.show()

    with open(os.path.join(dest_path, record_name), 'w') as fh:
        fh.write("{} detections in total\n".format(GM.detection_ct))
        fh.write("{} images without detections!\n".format(GM.no_detection_img_ct))
        for item in GM.no_detection_img_list:
            fh.write(item + "\n")

    print "{} detections in total".format(GM.detection_ct)
    print "{} images without detections!\n Their names saved to {}".format(
            GM.no_detection_img_ct, os.path.join(dest_path, record_name))

    
