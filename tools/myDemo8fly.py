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
import skimage
import base64
import requests
from StringIO import StringIO

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


def get_image_from_url(image_url):
    if img_url[:4] != "http":
        img_str = img_url.decode('base64')
    ## Image with Houzz url or an arbitrary external image url
    else:
        response = requests.get(img_url)
        if response.status_code == 200:
            img_str = response.content
        else:
            print "can not download image from url, status_code:", response.status_code, img_url
            return None
    img = skimage.io.imread(StringIO(img_str))
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    opencv_img_object = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return opencv_img_object


def demo(ct, net, image_url):
    """Detect object classes in an image"""

    # Load the demo image
    #im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
    print "count", ct

    #if not os.path.exists(image_name):
        #print "Invalid file, nonexistent: ", image_name
        #return
    #size_thr = 40000 #40k bytes threshold
    #img_file_size = os.stat(image_name).st_size
    #if img_file_size < size_thr:
        #print "Invalid file, size too small: ", image_name, img_file_size
        #return
    #im = cv2.imread(image_name)
    
    #timer0 = Timer()
    #timer0.tic()
    try:
        im = get_image_from_url(image_url)
    except:
        print "Invalid file, nonexistent: ", image_url
        return

    dim_thr = 350 * 480
    if im.shape[0] * im.shape[1] < dim_thr:
        print "Invalid file, size too small: ", im.shape
        return
    #print ("Downloading the image took {:.3f}s ".format(timer0.toc() ))

    # Detect all object classes and regress object bounds

    #timer1 = Timer()
    #timer1.tic()
    scores, boxes = im_detect(net, im)
    #print ('Detection took {:.3f}s for '
           #'{:d} object proposals').format(timer1.toc(), boxes.shape[0])

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
        
        #apply threshold at early stage to reduce nms compuation
        pre_filter_ind = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[pre_filter_ind, :]
        cls_scores = cls_scores[pre_filter_ind]

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
    if all_dets is None:
        print "No detections at all!"
        #GM.no_detection_img_ct += 1 
        #GM.no_detection_img_list.append(image_name)
        return
    all_keep = nms(all_dets, GM.nms_thresh_all) #nms among all classes
    all_dets = all_dets[all_keep, :]
    all_cls = np.array(all_cls)[all_keep]

    above_thresh = np.where(all_dets[:, -1] >= CONF_THRESH)[0]
    select_cls = all_cls[above_thresh]
    select_dets = all_dets[above_thresh, :]
    #vis_detections("/root/packages/py-faster-rcnn/tools/output", "test1", im, select_cls, select_dets, thresh=0.2)
    if len(above_thresh) == 0:
        #print image_name, "Nothing above threshold"
        print "Nothing above threshold"
    else:
        for i in range(select_dets.shape[0]):
            bbox = select_dets[i, :4]
            score = select_dets[i, -1]
            x = int(bbox[0])
            y = int(bbox[1])
            wid = int(bbox[2] - bbox[0]) + 1
            hei = int(bbox[3] - bbox[1]) + 1
            print image_url, x, y, wid, hei, select_cls[i], score




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
    parser.add_argument('--list', dest='img_list', help='Image list')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    #usage:
    #python tools/myDemo8.py --gpu 0  --conf 0.1 --list tools/all_space_photos_03102016_filelist_aa 2>&1 | tee tools/rst_all_space_photos_03102016_ab.txt

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    GM.conf_thresh = args.conf_thresh

    #img_list = "/root/packages/py-faster-rcnn/tools/all_spaces_6p8m.txt"
    #img_list = "/home/mark/space_photos_featured_02292016_filelist"
    #img_list = os.path.join(cfg.ROOT_DIR, 'tools', 'all_space_photos_03102016_filelist_aa')
    #img_list = os.path.join(cfg.ROOT_DIR, 'tools', 'all_space_photos_03102016_filelist_ab')
    img_list = args.img_list
    #model_path = "/root/packages/py-faster-rcnn/output/faster_rcnn_alt_opt/houzzdata0_train/"
    model_path = "/root/packages/py-faster-rcnn/output/faster_rcnn_end2end/houzzdata1_train/"
    #model_basename = "vgg16_fast_rcnn_stage2_iter_80000_train_40000.caffemodel"
    model_basename = "vgg16_faster_rcnn_n0016_iter_90200_svd3_fc6_256_fc7_64.caffemodel"

    #net configuration
    my_net =  ('VGG16', model_basename)
    #prototxt = os.path.join(cfg.ROOT_DIR, 'models', my_net[0],
                            #'faster_rcnn_alt_opt', 'faster_rcnn_test_vgg16_c138.pt')
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', my_net[0],
                            'faster_rcnn_end2end_houzzdata1', 'faster_rcnn_end2end_test_vgg16_svd3_c138.prototxt')
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
            img_url = line.strip()
            try:
                demo(cnt, net, img_url)
            except:
                print "Invalid file, unknown error: ", img_url
            if (args.end > 0 and cnt >= args.end):
                break


    
