#!/usr/bin/env python
import _init_paths
from fast_rcnn.config import cfg
from utils.timer import Timer
import os, sys
import argparse
import time
from IPython import embed
import subprocess





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
    #parser.add_argument('--model', dest='caffemodel', help='Model for test stage')
    parser.add_argument('--conf', dest='conf_thresh', help='Confidence threshold to use [0.5]',
                        default = 0, type=float)
    parser.add_argument('--scl', dest='scale_num', help='Number of scales to use [1]',
                        default=1, type=int)
    #parser.add_argument('--proto', dest='prototxt', help='Prototxt for test stage')
    #parser.add_argument('--list', dest='img_list', help='List of test images')
    #parser.add_argument('--imgpath', dest='img_path', help='Path to test images')
    #parser.add_argument('--rec', dest='record_name', help='Detetion record name')
    #parser.add_argument('--destrt', dest='dest_path_root', help='Result path')
    #parser.add_argument('--ext', dest='img_ext', help='Image extension name')


    args = parser.parse_args()

    return args




if __name__ == '__main__':
    #usage(--conf will draw figures):
    #python tools/run_batch_detect.py --gpu 0  --conf 0.2 --scl 3
    #python tools/run_batch_detect.py --gpu 0  --conf 0.2 --scl 4
    #python tools/run_batch_detect.py --gpu 0  --conf 0.2 --scl 1

    args = parse_args()
    gpu_id = args.gpu_id 
    conf_thresh = args.conf_thresh
    scale_num = args.scale_num
    record_name = "detection_record.txt"
    img_ext = ".jpg"

    #file for all image names
    #img_list = "./data/HouzzDataCollection/HouzzData0/ImageLists/val.txt"
    #img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/val.txt"
    #img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/val_nosquare.txt"
    #img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/val_nosquare3.txt"
    #img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/test.txt"
    img_list = "./data/HouzzDataCollection/HouzzData1/ImageLists/temp.txt"

    #path to all images
    #img_path = "./data/HouzzDataCollection/HouzzData0/Images/"
    img_path = "./data/HouzzDataCollection/HouzzData1/Images/"

    #image extension name
    #img_ext = "" #for lists that already contain extention in the image names
    #img_ext = ".jpg" 
    
    #output path for resulting images and report
    #dest_path = "./work_data/faster-rcnn/VGG16/dets_end2end_n0016_train_nosquare_i90200_houzzdata1_val_nosquare_scl{}_max{}_earlyTh/t{:0.1f}/".format(cfg.TEST.SCALES[0], cfg.TEST.MAX_SIZE, GM.conf_thresh)
    #dest_path = "./work_data/faster-rcnn/VGG16/dets_end2end_n0019_train_nosquare_i90000_houzzdata1_val_nosquare_mulscl{}/t{:0.1f}/".format(len(GM.test_scale_list), GM.conf_thresh)
    #dest_path_root = "./work_data/faster-rcnn/VGG16/dets_end2end_n0019_train_nosquare_i90000_houzzdata1_val_nosquare"
    #dest_path_root = "./work_data/faster-rcnn/RESNET/dets_end2end_n0047_train_nosquare_i200000_houzzdata1_val_nosquare"
    dest_path_root = "./work_data/faster-rcnn/VGG16/dets_end2end_n0016_train_nosquare_i90200_svd3_houzzdata1_val_nosquare"

    #path to models
    #model_path = "./output/faster_rcnn_alt_opt/voc_2007_trainval/"
    #model_path = "./output/faster_rcnn_end2end/houzzdata0_train/"
    model_path = "./output/faster_rcnn_end2end/houzzdata1_train/"

    #name of the model to use
    model_basename = "vgg16_faster_rcnn_n0016_iter_90200_svd3_fc6_256_fc7_64.caffemodel"
    #model_basename = "vgg16_faster_rcnn_n0017_iter_90000.caffemodel"
    #model_basename = "vgg16_faster_rcnn_n0017_iter_90000_svd3_fc6_256_fc7_64.caffemodel"
    #model_basename = "vgg16_faster_rcnn_n0019_iter_90000.caffemodel"
    #model_basename = "resnet50_faster_rcnn_n0029_iter_150000.caffemodel"
    #model_basename = "resnet50_faster_rcnn_n0047_iter_200000.caffemodel"

    #net configuration
    #my_net =  ('ZF', model_basename)
    my_net =  ('VGG16', model_basename)
    #my_net =  ('RESNET', model_basename)

    #prototxt = os.path.join(cfg.ROOT_DIR, 'models', my_net[0],
                            #'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #prototxt = os.path.join(cfg.ROOT_DIR, 'models', my_net[0],
                            #'faster_rcnn_end2end_houzzdata1', 'test.prototxt')
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', my_net[0],
                            'faster_rcnn_end2end_houzzdata1', 'faster_rcnn_end2end_test_vgg16_svd3_c138.prototxt')
    #prototxt = os.path.join(cfg.MODELS_DIR, 'models', my_net[0],
                            #'faster_rcnn_end2end_houzzdata1_dp', 'test.prototxt')
    #prototxt = os.path.join(cfg.MODELS_DIR, my_net[0],
                            #'faster_rcnn_end2end_houzzdata1', 'test50.prototxt')

    caffemodel = os.path.join(model_path, model_basename)
    #run_detection(dest_path, prototxt, caffemodel)
    
    input_args =  "--gpu {gpu_id} --model {caffemodel} --conf {conf_thresh} \
            --scl {scale_num} --proto {prototxt} --list {img_list} \
            --imgpath {img_path} --rec {record_name} --destrt {dest_path_root} \
            --ext {img_ext}".format(\
            gpu_id=gpu_id, caffemodel=caffemodel, conf_thresh=conf_thresh, \
            scale_num=scale_num, prototxt=prototxt, img_list=img_list,\
            img_path=img_path, record_name=record_name, dest_path_root=dest_path_root,\
            img_ext=img_ext)
    
    try:
        retcode = subprocess.call("python tools/batch_detect.py " + input_args, shell=True)
        if retcode < 0:
            print >> sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >> sys.stderr, "Child returned ", retcode
    except OSError as e:
        print >> sys.stderr, "Execution failed:", e
