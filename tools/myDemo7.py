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
        self.nms_thresh_all = 0.5


GM = global_meta()


def get_classes():
    #cate_table_file = "/home/ubuntu/work/data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_table_merged_modified.txt"
    #cate_table_file = "/home/ubuntu/work/data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_table_purged.txt"
    cate_table_file = "/root/packages/py-faster-rcnn/tools/cat_table_purged.txt"
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
            wid = int(bbox[2] - bbox[0])
            hei = int(bbox[3] - bbox[1])
            fh.write("{:d} {:d} {:d} {:d} {:d} {:.4f}\n".format(x, y, wid, hei, class_inds[i], score))



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


def demo(ct, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    #print ('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #CONF_THRESH = 0.5
    CONF_THRESH = GM.conf_thresh
    #NMS_THRESH = 0.3
    all_dets = None
    all_cls = None
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #keep = nms(dets, NMS_THRESH) # nms in just this class
        keep = nms(dets, GM.nms_thresh_within_class) # nms in just this class
        #embed()
        dets = dets[keep, :]
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
    all_keep = nms(all_dets, GM.nms_thresh_all) #nms among all classes
    all_dets = all_dets[all_keep, :]
    all_cls = np.array(all_cls)[all_keep]
    if all_dets is None:
        print "No detections at all!"
        #GM.no_detection_img_ct += 1 
        #GM.no_detection_img_list.append(image_name)
        #return

    above_thresh = np.where(all_dets[:, -1] >= .2)[0]
    select_cls = all_cls[above_thresh]
    select_dets = all_dets[above_thresh, :]
    #vis_detections("/root/packages/py-faster-rcnn/tools/output", "test1", im, select_cls, select_dets, thresh=0.2)
    print "count", ct
    if len(above_thresh) == 0:
        print image_name, "Nothing above threshold"
    else:
        for i in range(select_dets.shape[0]):
            bbox = select_dets[i, :4]
            score = select_dets[i, -1]
            x = int(bbox[0])
            y = int(bbox[1])
            wid = int(bbox[2] - bbox[0])
            hei = int(bbox[3] - bbox[1])
            print image_name, x, y, wid, hei, select_cls[i], score

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--start', dest='start', help='start',
                        default=0, type=int)
    parser.add_argument('--end', dest='end', help='end',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--conf', dest='conf_thresh', help='Confidence threshold to use [0.5]',
                        default = 0, type=float)
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    #usage:
    #python tools/myDemo7.py --gpu 2  --conf 0.2

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    GM.conf_thresh = args.conf_thresh

    img_list = "/root/packages/py-faster-rcnn/tools/all_spaces_6p8m.txt"
    model_path = "/root/packages/py-faster-rcnn/output/faster_rcnn_alt_opt/houzzdata0_train/"
    model_basename = "vgg16_fast_rcnn_stage2_iter_80000_train_40000.caffemodel"

    #net configuration
    my_net =  ('VGG16', model_basename)
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', my_net[0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test_vgg16_c138.pt')
    caffemodel = os.path.join(model_path, my_net[1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    cnt = 0
    with open(img_list, 'r') as ins:
        if (args.start > 0):
            for line in ins:
                cnt += 1
                if (cnt == args.start):
                    break
        for line in ins:
            cnt += 1
            img_name = line.strip()
            demo(cnt, net, img_name)
            if (args.end > 0 and cnt >= args.end):
                break


    
