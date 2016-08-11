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
    #cate_table_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_table_merged_modified.txt"
    cate_table_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_table_purged.txt"
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




def demo(net, image_name, dest_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

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
        #vis_detections(im, cls, dets, dest_path, thresh=CONF_THRESH)
        if dets.shape[0] <= 0:
            continue
        if all_dets == None:
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
    if all_dets == None:
        print "No detections at all!"
        GM.no_detection_img_ct += 1 
        GM.no_detection_img_list.append(image_name)
        return

    #write all boxes to disk 
    image_basename = os.path.basename(image_name)
    boxfile_pathname = os.path.join(dest_path, image_basename[:-4] + '.txt')
    #write_detections(boxfile_pathname,  all_cls, all_dets)

    #threshold not set, do not visualize
    if CONF_THRESH <= 0:
        print "Threshold not set, exit"
        exit(1)
    
    #visualize boxes whose scores are above the threshold
    above_thresh = np.where(all_dets[:, -1] >= CONF_THRESH)[0]
    #embed()
    select_cls = all_cls[above_thresh]
    select_dets = all_dets[above_thresh, :]
    print '{} detections above the threshold {}'.format(len(select_cls), CONF_THRESH)
    if len(above_thresh) == 0:
        print "Nothing above threshold!"
        GM.no_detection_img_ct += 1 
        GM.no_detection_img_list.append(image_name)
        return
    else:
        #vis_detections(dest_path, image_basename, im, select_cls, select_dets, thresh=CONF_THRESH)
        write_detections(boxfile_pathname,  select_cls, select_dets)



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
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
    #Usage (no figures)
    #python tools/myDemo7_btsp.py --gpu 0  --conf 0.1

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    GM.conf_thresh = args.conf_thresh

    #file for all image names
    #img_list = "./data/HouzzDataCollection/HouzzData0/ImageLists/val.txt"
    #img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/val.txt"
    #img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/train34000+6000_firstquater.txt"
    img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/train34000+6000_secondquater.txt"

    #path to all images
    #img_path = "./data/HouzzDataCollection/HouzzData0/Images/"
    img_path = "./data/HouzzDataCollection/HouzzData1/Images/"

    #image extension name
    #img_ext = "" #for lists that already contain extention in the image names
    img_ext = ".jpg" 
    
    #output path for resulting images and report
    #dest_path = "./work_data/faster-rcnn/VGG16/dets_i136040_train68000_houzzdata1_val/t{:0.1f}/".format(GM.conf_thresh)
    #dest_path = "./work_data/faster-rcnn/VGG16/dets_i68000_houzzdata1_train34000_houzzdata1_val/t{:0.1f}/".format(GM.conf_thresh)
    #dest_path = "./work_data/faster-rcnn/VGG16/dets_end2end_i30000_train68000/t{:0.1f}/".format(GM.conf_thresh)
    #dest_path = "./work_data/faster-rcnn/VGG16/dets_end2end_i136000_houzzdata1_raw_train34000/t{:0.1f}/".format(GM.conf_thresh)
    #dest_path = "./work_data/faster-rcnn/VGG16/dets_end2end_i136000_houzzdata1_train34000_houzzdata1_val/t{:0.1f}/".format(GM.conf_thresh)
    #dest_path = "./work_data/faster-rcnn/VGG16/dets_end2end_t0000_houzzdata1_val/t{:0.1f}/".format(GM.conf_thresh)
    #dest_path = "./work_data/faster-rcnn/VGG16/dets_end2end_t0000_val/t{:0.1f}/".format(GM.conf_thresh)
    dest_path = "./work_data/faster-rcnn/VGG16/dets_i136040_train68000_houzzdata1_train34000+6000/t{:0.1f}/".format(GM.conf_thresh)

    #file to record some stats for each image 
    record_name = "detection_record.txt"

    #path to models
    #model_path = "./output/faster_rcnn_alt_opt/voc_2007_trainval/"
    model_path = "./output/faster_rcnn_alt_opt/houzzdata0_train/"
    #model_path = "./output/faster_rcnn_alt_opt/houzzdata1_train/"
    #model_path = "./output/faster_rcnn_alt_opt_g1/houzzdata0_train/"
    #model_path = "./output/faster_rcnn_alt_opt_g2/houzzdata0_train/"
    #model_path = "./output/faster_rcnn_end2end/houzzdata0_train/"
    #model_path = "./output/faster_rcnn_end2end/houzzdata1_train/"

    #name of the model to use
    #model_basename = "VGG16_faster_rcnn_final.caffemodel"
    #model_basename = "zf_fast_rcnn_stage2_iter_60000_train_60000.caffemodel"
    #model_basename = "zf_fast_rcnn_stage2_iter_136020_train_68000.caffemodel"
    #model_basename = "zf_faster_rcnn_iter_2000.caffemodel"
    #model_basename = "vgg16_fast_rcnn_stage2_iter_136020-90000_train_68000.caffemodel"
    #model_basename = "vgg16_fast_rcnn_stage2_iter_68000_houzzdata1_raw_train_34000.caffemodel"
    #model_basename = "vgg16_faster_rcnn_iter_68000.caffemodel"
    model_basename = "vgg16_fast_rcnn_stage2_iter_136040.caffemodel"
    #model_basename = "vgg16_fast_rcnn_stage2_iter_68000_houzzdata1_train_34000.caffemodel"
    #model_basename = "vgg16_faster_rcnn_iter_136000_houzzdata1_train_34000.caffemodel"
    #model_basename = "vgg16_faster_rcnn_t0000_iter_136000.caffemodel"
    #net configuration
    #my_net =  ('ZF', model_basename)
    my_net =  ('VGG16', model_basename)

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', my_net[0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    
    #prototxt = os.path.join(cfg.ROOT_DIR, 'models', my_net[0],
                            #'faster_rcnn_end2end', 'test.prototxt')

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
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

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)


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
            demo(net, img_name, dest_path)
            t2 = time.time()
            print "time cost for this image: {:.2f}s".format(t2 - t1)
            print "{} images without detections!".format(GM.no_detection_img_ct)
            timeCostTotal += t2 - t1
            ct += 1
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

    
