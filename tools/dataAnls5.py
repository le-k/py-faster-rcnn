import _init_paths
import numpy as np
import operator
import os
from IPython import embed
import difflib
import random
import boto
import glob
import re
import PIL
import cv2
import matplotlib.pyplot as plt
import time
import socket
import requests
import sys
import shutil
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg

#hostname =  socket.gethostname()
#if hostname[:3] == "ip-":
    ##aws ec2 instance
    #from boto.s3.key import Key
    #conn = boto.connect_s3()
    #bucket_list = [conn.get_bucket("st00%d.houzz.com" % i) for i in range(8)]
    #bucket_map = {'0':0, '1':0, '2':1, '3':1, '4':2, '5':2, '6':3, '7':3, '8':4, '9':4, 'a':5, 'b':5, 'c':6, 'd':6, 'e':7, 'f':7}
#else:
    #pass


def load_cate_merge_map(filepath):
    assert os.path.exists(filepath)
    cate_merge_map = {}
    with open(filepath, 'r') as fh:
        for line in fh:
            vals = line.strip().split()
            cate_merge_map[vals[0]] = vals[1]

    return cate_merge_map



def merge(cate_merge_map, orgl_file, dst_file, fixed_target_value=None):
    with open(dst_file, 'w') as fw:
        with open(orgl_file, 'r') as fr:
            for line in fr:
                vals = line.split('\t')
                if fixed_target_value is not None:
                    vals[-3] = fixed_target_value
                else:
                    if vals[-3] in cate_merge_map:
                        vals[-3] = cate_merge_map[vals[-3]]
                fw.write( '\t'.join(vals))
        
             

def merge_cate():
    ref_train_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039.txt"
    ref_val_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_holdout2000.txt"
    ref_test_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_testset5000.txt"
    ref_all_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015.txt"
    dst_train_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039_cate_merged.txt"
    dst_val_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_holdout2000_cate_merged.txt"
    dst_test_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_testset5000_cate_merged.txt"
    dst_all_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_cate_merged.txt"

    cate_merge_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_merge_map.txt"
    
    cate_merge_map = load_cate_merge_map(cate_merge_file)

    merge(cate_merge_map, ref_train_file, dst_train_file)
    merge(cate_merge_map, ref_val_file, dst_val_file)
    merge(cate_merge_map, ref_test_file, dst_test_file)
    merge(cate_merge_map, ref_all_file, dst_all_file)



def show_merge_names_and_counts():
    print "Show category names for a merge map"
    cate_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/categories.txt"
    merge_map_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_merge_map.txt" 
    orgl_cate_dstb_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cate_dstb_tag_boxes_11192015.txt"
    merged_cate_dstb_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cate_dstb_merged_all.txt"
    merge_map_show_names_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_merge_map_show_names.txt" 
   
    def get_dstb(filename):
        cate_dstb = {}
        with open(filename, 'r') as fh:
            for line in fh:
                vals = line.strip().split('\t')
                cate_dstb[vals[1]] = int(vals[0]) 
        return cate_dstb

    orgl_cate_dstb = get_dstb(orgl_cate_dstb_file)
    merged_cate_dstb = get_dstb(merged_cate_dstb_file)
    cate_id_to_name = get_cate_name_map(cate_file)
    all_list = []
    #get categories shown in the merge map
    with open(merge_map_file, 'r') as fr:
        for line in fr:
            vals = line.strip().split()
            from_cate_id = vals[0]
            to_cate_id = vals[1]
            all_list.append( (from_cate_id, to_cate_id) )
    all_from_cate_id_list, _ = zip(*all_list)
    all_from_cate_id_set = set(all_from_cate_id_list)
    #get all other categories
    for cate_id in cate_id_to_name.keys():
        if cate_id not in all_from_cate_id_set:
            all_list.append( (cate_id, cate_id) )

    sorted_list = sorted(all_list, key=lambda x: (int(x[1]), int(x[0])) ) #sort by two keys
    with open(merge_map_show_names_file, 'w') as fw:
        for from_cate_id, to_cate_id in sorted_list:
            from_cate_name = cate_id_to_name[from_cate_id]
            to_cate_name = cate_id_to_name[to_cate_id]
            from_cate_ct = 0
            to_cate_ct = 0
            if from_cate_id in orgl_cate_dstb:
                from_cate_ct = orgl_cate_dstb[from_cate_id]
            if to_cate_id in merged_cate_dstb:
                to_cate_ct = merged_cate_dstb[to_cate_id]
            fw.write("{} ({})\t{} ({})\t\t{}\t\t\t\t{}\n".format(from_cate_id, from_cate_ct, to_cate_id, to_cate_ct, from_cate_name, to_cate_name))
    


def find_space_overlap():
    spaceCt = {}
    with open('./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_testset5000.txt', 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            space_id = vals[1]
            if space_id in spaceCt:
                spaceCt[space_id] += 1
            else:
                spaceCt[space_id] = 1

    overlapCt = {}
    with open('./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039.txt', 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            space_id = vals[1]
            if space_id in spaceCt:
                if space_id in overlapCt:
                    overlapCt[space_id][1] += 1
                else:
                    overlapCt[space_id] = [spaceCt[space_id], 1]
    sorted_overlap = sorted(overlapCt.items(), key = (lambda t: t[1][1]), reverse=True)
    print type(sorted_overlap)
    with open('./data/HouzzDataCollection/HouzzData0/OrigBoxList/space_overlap_test_train.txt', 'w') as fh:
        for item in sorted_overlap:
            print "{}\t\t{}\t\t{}\n".format(item[0], item[1][0], item[1][1])
            fh.write("{}\t\t{}\t\t{}\n".format(item[0], item[1][0], item[1][1]))



def get_cate_dstb_from_box_list(filename):
    cateCt = {}
    with open(filename, 'r') as fh:
        tot = 0
        for line in fh:
            tot += 1
            vals = line.strip().split('\t')
            cate_id = vals[-3] #target category id
            if cate_id in cateCt:
                cateCt[cate_id] += 1
            else:
                cateCt[cate_id] = 1
    return cateCt


def get_cate_name_map(filename):
    cate_id_to_name = {}
    with open(filename, 'r') as fh:
        tot = 0
        for line in fh:
            tot += 1
            vals = line.strip().split('\t')
            cate_id = vals[-3]
            cate_name = vals[-1]
            cate_id_to_name[cate_id] = cate_name
    return cate_id_to_name


#def cate_dstb_from_box_list():
    #catefile = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/categories.txt"
    ##dataset = "_testset5000"
    ##dataset = "_holdout2000"
    ##dataset = "_trainset94039"
    ##boxfile = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015{}.txt".format(dataset)
    ##rstfile = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/categories{}.txt".format(dataset)
    ##boxfile = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039_cate_merged.txt"
    ##rstfile = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/categories_trainset94039_cate_merged.txt"
    ##boxfile = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015.txt"
    ##rstfile = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cate_dstb_tag_boxes_11192015.txt"
    #boxfile = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_cate_merged.txt"
    #rstfile = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cate_dstb_tag_boxes_11192015_cate_merged.txt"

    #print "box file : {}".format(boxfile)

    #cateCt = get_cate_dstb_from_box_list(boxfile)
    #cate_id_to_name = get_cate_name_map(catefile)    
    #sorted_cateCt = sorted(cateCt.items(), key=operator.itemgetter(1), reverse=True)
    #with open(rstfile, 'w') as fh:
        #print "cate num: {}\n".format(len(cateCt.keys()))
        ##fh.write("cate num: {}\n".format(len(cateCt.keys())))
        #for cate_id, ct in sorted_cateCt:
            #print "{}\t{}\t{}\n".format(ct, cate_id, cate_id_to_name[cate_id] ),
            #fh.write("{}\t{}\t{}\n".format(ct, cate_id, cate_id_to_name[cate_id] ))

    #return
        


def partition_merged_sets():
    print "Partition the boxes with merged categories into train, validation, and test, without overlapping space_ids"
    ref_train_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039_cate_merged.txt"
    ref_val_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_holdout2000_cate_merged.txt"
    ref_test_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_testset5000_cate_merged.txt"
    #test_space_num_to_set = 4800
    #val_space_num_to_set = 1800
    dest_path = "./data/HouzzDataCollection/HouzzData0/imageListsAll/cate_merged"
    train_index_file = "train.txt"
    val_index_file = "val.txt"
    test_index_file = "test.txt"

    partition_sets(ref_train_file, ref_val_file, ref_test_file, dest_path, train_index_file, val_index_file, test_index_file)



def partition_sets(ref_train_file, ref_val_file, ref_test_file, dest_path, train_index_file, val_index_file, test_index_file):
    test_space_dict = {}
    with open(ref_test_file, 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            space_id = vals[1]
            if space_id in test_space_dict:
                test_space_dict[space_id] += 1
            else:
                test_space_dict[space_id] = 1
    print "test spaces : {}".format( len(test_space_dict.keys()) ) 
    #extra = test_space_dict.items()[test_space_num_to_set:]
    #test_space_dict = dict(test_space_dict.items()[:test_space_num_to_set])
    #print "test spaces after truncate: {}".format( len(test_space_dict.keys()) ) 

    val_space_dict = {}
    with open(ref_val_file, 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            space_id = vals[1]
            if space_id in test_space_dict:
                pass
            else:
                if space_id in val_space_dict:
                    val_space_dict[space_id] += 1
                else:
                    val_space_dict[space_id] = 1
    #extra.extend(val_space_dict.items()[val_space_num_to_set:])
    #val_space_dict = dict( val_space_dict.items()[:val_space_num_to_set] )
    print "val spaces: {}".format( len(val_space_dict.keys()))

    train_space_dict = {}
    with open(ref_train_file, 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            space_id = vals[1]
            if (space_id in test_space_dict) or (space_id in val_space_dict):
                pass
            else:
                if space_id in train_space_dict:
                    train_space_dict[space_id] += 1
                else:
                    train_space_dict[space_id] = 1

    print "train spaces: {}".format( len(train_space_dict.keys()))

    with open(os.path.join(dest_path, train_index_file), 'w') as fh:
        for space in train_space_dict.keys():
            fh.write(space + '\n')

    with open(os.path.join(dest_path, val_index_file), 'w') as fh:
        for space in val_space_dict.keys():
            fh.write(space + '\n')

    with open(os.path.join(dest_path, test_index_file), 'w') as fh:
        for space in test_space_dict.keys():
            fh.write(space + '\n')





def update_dict(new_key, my_dict):
    if new_key in my_dict:
        my_dict[new_key] += 1
    else:
        my_dict[new_key] = 1




def get_cate_dstb_for_set(space_to_boxes, cate_id_to_name, index_path, dest_path):
    cate_dstb = {}
    with open(index_path, 'r') as fh:
        for line in fh:
            ind = line.strip()
            for box in space_to_boxes[ind]:
                update_dict(box[-3], cate_dstb) 
    sorted_dstb = sorted(cate_dstb.items(), key = operator.itemgetter(1), reverse=True) 
    with open(dest_path, 'w') as fh:
        for cid, ct in sorted_dstb:
            #fh.write( "{}\t{}\n".format(cate, ct) )
            fh.write("{}\t{}\t{}\n".format(ct, cid, cate_id_to_name[cid] ))
        



def cate_dstb_for_sets():
    print "Find category distribution for train, validation, test sets, or all samples, according to the image list"
    #merge_cate = True
    merge_cate = False
    purge_cate = True

    dataset_version = 0
    #dataset_version = 1

    catefile = "./data/HouzzDataCollection/HouzzData{}/OrigBoxList/categories.txt".format(dataset_version)
    #index_dir = "./data/HouzzDataCollection/HouzzData0/imageListsAll/cate_merged"
    index_dir = "./data/HouzzDataCollection/HouzzData{}/imageListsAll/imageLists_cate_purged".format(dataset_version)
    train_index_file = "train.txt"
    #val_index_file = "val.txt"
    #test_index_file = "test.txt"
    #all_index_file = "all.txt"
    dest_dir = "./data/HouzzDataCollection/HouzzData{}/OrigBoxList/".format(dataset_version)
    all_box_list = "./data/HouzzDataCollection/HouzzData{}/OrigBoxList/tag_boxes_11192015.txt".format(dataset_version)
    #all_box_list = "./data/HouzzDataCollection/HouzzData{}/OrigBoxList/image_tag_bounds_01122016.txt".format(dataset_version)
    dest_prefix = "cate_dstb_"
    if merge_cate:
        all_box_list = all_box_list[:-4] + "_cate_merged.txt"
        dest_prefix =  dest_prefix + "merged_"
    elif purge_cate:
        all_box_list = all_box_list[:-4] + "_cate_purged.txt"
        dest_prefix =  dest_prefix + "purged_"
    else:
        print "Neither merge or purge"

    cate_id_to_name = get_cate_name_map(catefile)    
    space_to_boxes = {}
    with open(all_box_list, 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            space_id = vals[1]
            if space_id in space_to_boxes:
                space_to_boxes[space_id].append(vals)
            else:
                space_to_boxes[space_id] = [vals]

    get_cate_dstb_for_set(space_to_boxes, cate_id_to_name, os.path.join(index_dir, train_index_file), os.path.join(dest_dir, dest_prefix + train_index_file))
    #get_cate_dstb_for_set(space_to_boxes, cate_id_to_name, os.path.join(index_dir, val_index_file), os.path.join(dest_dir, dest_prefix + val_index_file))
    #get_cate_dstb_for_set(space_to_boxes, cate_id_to_name, os.path.join(index_dir, test_index_file), os.path.join(dest_dir, dest_prefix + test_index_file))
    #get_cate_dstb_for_set(space_to_boxes, cate_id_to_name, os.path.join(index_dir, all_index_file), os.path.join(dest_dir, dest_prefix + all_index_file))
     
    

def split_box_list_of_merged_cates(merge_cate = True):
    all_box_list = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015.txt"
    dest_dir = "./data/HouzzDataCollection/HouzzData0/annotationsAll/original/" 
    if merge_cate:
        all_box_list = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_cate_merged.txt"
        dest_dir = "./data/HouzzDataCollection/HouzzData0/annotationsAll/cate_merged/"
    print "all boxes: {}".format(all_box_list)
    print "dest dir: {}".format(dest_dir)
    split_box_list_by_space(all_box_list, dest_dir)



#def split_box_list_by_space(merge_cate = True):
def split_box_list_by_space(all_box_list, dest_dir):
    spaceCt = {}
    #all_box_list = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015.txt"
    #dest_dir = "./data/HouzzDataCollection/HouzzData0/annotationsAll/original/" 
    #if merge_cate:
        #all_box_list = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_cate_merged.txt"
        #dest_dir = "./data/HouzzDataCollection/HouzzData0/annotationsAll/cate_merged/"
    #print "all boxes: {}".format(all_box_list)
    #print "dest dir: {}".format(dest_dir)

    with open(all_box_list, 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            space_id = vals[1]
            if space_id in spaceCt:
                spaceCt[space_id].append(line)
            else:
                spaceCt[space_id] = [line]
    ct = 0
    max_box = 0
    for space_id, boxes in spaceCt.items():
        ct += 1 
        box_ct = 0
        #with open(os.path.join(dest_dir, '{}.txt'.format(space_id)), 'w') as fh:
        with open(os.path.join(dest_dir, '{}.txt'.format(space_id)), 'a') as fh:
            #use append because some boxes may be added instead of overwriting ones
            for box in boxes:
                fh.write(box) #each line already has a return
                box_ct += 1
            if box_ct > max_box:
                max_box = box_ct 
    print "{} spaces  in total !".format(ct)
    print "max box number: {}".format(max_box)
    return 





def sort_all_boxes():
    path1 = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015.txt"
    path2 = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11242015.txt"
    list1 = []
    list2 = []
    with open(path1, 'r') as fh:
        list1 = fh.read().splitlines() 
    list1 = sorted(list1)
    with open(path2, 'r') as fh:
        list2 = fh.read().splitlines()
    list2 = sorted(list2)
    print "File 1 equivalent to file 2 ? {}".format( list1 == list2)
    ct = 0
    for k in range(len(list1)):
        if list1[k] != list2[k]:
            ct += 1
    print "total different rows: {}".format(ct)
    #embed()
    for line in difflib.unified_diff(list1, list2, fromfile = "list1.py", tofile = "list2.py"):
        print line 




    

def get_sample_training_images():
    index_dir = "./data/HouzzDataCollection/HouzzData0/imageListsAll/cate_merged"
    orgl_file = "train.txt"
    sample_size = 1000

    sample_file = "train{}.txt".format(sample_size)
    lines = []
    with open(os.path.join(index_dir, orgl_file), 'r') as fh:
        lines = fh.readlines()
    random.shuffle(lines)            
    sample_lines = lines[:sample_size]
    with open(os.path.join(index_dir, sample_file), 'w') as fh:
        for line in sample_lines:
            fh.write(line)




def download_images():
    index_dir = "./data/HouzzDataCollection/HouzzData0/imageListsAll/imageLists_cate_merged"
    #list_file = "train1000.txt"
    #list_file = "val.txt"
    list_file = "all.txt"
    #annotation_dir = "./data/HouzzDataCollection/HouzzData0/annotationsAll/cate_merged/"
    annotation_dir = "./data/HouzzDataCollection/HouzzData0/annotationsAll/cate_merged/"
    images_dir = "./data/HouzzDataCollection/HouzzData0/Images/"
    
    download_images_general(index_dir, list_file, annotation_dir, images_dir)




def download_images_for_new_data():
    index_dir = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged"
    #list_file = "all_01122016.txt"
    #list_file = "all_01272016.txt"
    list_file = "all_02052016.txt"
    annotation_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged/"
    images_dir = "./data/HouzzDataCollection/HouzzData1/Images/"
    download_images_general(index_dir, list_file, annotation_dir, images_dir)





def check_oldnew_overlap_and_get_new_train_list():
    src0_file_all  = "./data/HouzzDataCollection/HouzzData0/imageListsAll/imageLists_cate_purged/all.txt"
    src0_file_val = "./data/HouzzDataCollection/HouzzData0/imageListsAll/imageLists_cate_purged/val.txt"
    src0_file_test = "./data/HouzzDataCollection/HouzzData0/imageListsAll/imageLists_cate_purged/test.txt"

    src1_file_all = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/all.txt"
    dst_file_train = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/train_orig.txt"

    set0_all = set()
    set0_val = set()
    set0_test = set()
    overlap_all_list = []
    overlap_valtest_list = []
    new_train_list = []

    with open(src0_file_all, 'r') as fr0:
        for line in fr0:
            set0_all.add(int(line.strip()))
    with open(src0_file_val, 'r') as fr1:
        for line in fr1:
            set0_val.add(int(line.strip()))
    with open(src0_file_test, 'r') as fr2:
        for line in fr2:
            set0_test.add(int(line.strip()))

    with open(src1_file_all, 'r') as fr3:
        for line in fr3:
            space_id = int(line.strip())
            if space_id in set0_all:
                overlap_all_list.append(space_id)
            if space_id in set0_val or space_id in set0_test: 
                #overlapping with old val or test data, discard
                overlap_valtest_list.append(space_id)
                continue
            else:
                new_train_list.append(space_id)
            
    #print "All to all, overlapping space_ids:"
    #for item in overlap_list:
        #print item, ', ',
    print '\n'
    print "All to all, {} overlapping space_ids!".format(len(overlap_all_list))

    print "All to val and test, {} overlapping space_ids!".format( len(overlap_valtest_list) )

    print "Generating new train list.."
    import random
    random.shuffle(new_train_list)
    new_train_content = '\n'.join( [ str(space_id) for space_id in new_train_list ] )
    with open(dst_file_train, 'w') as fw:
        fw.write(new_train_content)
    print "Done!"
   


    
def find_space_overlap_houzzdata1_different_dates():
    spaceCt = {}
    with open('./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01122016.txt', 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            space_id = vals[1]
            if space_id in spaceCt:
                spaceCt[space_id] += 1
            else:
                spaceCt[space_id] = 1

    overlapCt = {}
    with open('./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01272016.txt', 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            space_id = vals[1]
            if space_id in spaceCt:
                if space_id in overlapCt:
                    overlapCt[space_id][1] += 1
                else:
                    overlapCt[space_id] = [spaceCt[space_id], 1]
    sorted_overlap = sorted(overlapCt.items(), key = (lambda t: t[1][1]), reverse=True)
    print type(sorted_overlap)
    with open('./data/HouzzDataCollection/HouzzData1/OrigBoxList/space_overlap_between_01122016_and_01272016.txt', 'w') as fh:
        for item in sorted_overlap:
            print "{}\t\t{}\t\t{}\n".format(item[0], item[1][0], item[1][1])
            fh.write("{}\t\t{}\t\t{}\n".format(item[0], item[1][0], item[1][1]))




def check_oldnew_overlap_and_get_new_train_val_test_list():
    src0_file_all  = "./data/HouzzDataCollection/HouzzData0/imageListsAll/imageLists_cate_purged/all.txt"
    src0_file_val = "./data/HouzzDataCollection/HouzzData0/imageListsAll/imageLists_cate_purged/val.txt"
    src0_file_test = "./data/HouzzDataCollection/HouzzData0/imageListsAll/imageLists_cate_purged/test.txt"

    #src1_file_prev = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/all_01122016.txt"
    src1_file_prev = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/all_01122016+01272016.txt"
    src1_file_all = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/all_02052016.txt"

    #dst_file_val = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/train_orig.txt"
    dst_file_val = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/val_orig.txt"
    dst_file_test = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/test_orig.txt"
    dst_file_train = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/train_orig.txt"

    #val_tot = 2000
    #test_tot = 5000
    val_tot = 0
    test_tot = 0

    set0_all = set()
    #set0_val = set()
    #set0_test = set()
    set1_prev = set()
    overlap_all_list = []
    overlap_prev_list = []
    new_trainvaltest_list = []

    with open(src0_file_all, 'r') as fr0:
        for line in fr0:
            set0_all.add(int(line.strip()))
    #with open(src0_file_val, 'r') as fr1:
        #for line in fr1:
            #set0_val.add(int(line.strip()))
    #with open(src0_file_test, 'r') as fr2:
        #for line in fr2:
            #set0_test.add(int(line.strip()))
    with open(src1_file_prev, 'r') as fr3:
        for line in fr3:
            set1_prev.add(int(line.strip()))

    with open(src1_file_all, 'r') as fr3:
        for line in fr3:
            space_id = int(line.strip())
            if space_id in set0_all:
                overlap_all_list.append(space_id)
            elif space_id in set1_prev: 
                #overlapping with old val or test data, discard
                overlap_prev_list.append(space_id)
            else:
                new_trainvaltest_list.append(space_id)
    #print "All to all, overlapping space_ids:"
    #for item in overlap_list:
        #print item, ', ',
    print '\n'
    print "All to all, {} overlapping space_ids!".format(len(overlap_all_list))
    print "All to prev, {} overlapping space_ids!".format( len(overlap_prev_list) )
    print "Generating new train, val, and test list.."
    import random
    random.shuffle(new_trainvaltest_list)
    print "New train val and test total: {}".format( len(new_trainvaltest_list) )
    new_val_list = new_trainvaltest_list[:val_tot] 
    new_test_list = new_trainvaltest_list[val_tot:(val_tot+test_tot)]
    new_train_list = new_trainvaltest_list[(val_tot+test_tot):]
    if new_val_list != []:
        print "write new val set"
        new_val_content = '\n'.join( [ str(space_id) for space_id in new_val_list ] )
        with open(dst_file_val, 'w') as fw:
            fw.write(new_val_content)
    if new_test_list != []:
        print "write new test set"
        new_test_content = '\n'.join( [ str(space_id) for space_id in new_test_list ] )
        with open(dst_file_test, 'w') as fw:
            fw.write(new_test_content)
    if new_train_list != []:
        print "write new train set"
        new_train_content = '\n'.join( [ str(space_id) for space_id in new_train_list ] )
        with open(dst_file_train, 'w') as fw:
            fw.write(new_train_content)
    print "Done!"






def download_images_general(index_dir, list_file, annotation_dir, images_dir):
    print "Start to download images according to this list: ", list_file
    ct = 0
    success_ct = 0
    fail_ct = 0
    fail_list = []
    img_src = ""
    hostname =  socket.gethostname()
    if hostname[:3] == "ip-":
        #aws ec2 instance
        from boto.s3.key import Key
        conn = boto.connect_s3()
        bucket_list = [conn.get_bucket("st00%d.houzz.com" % i) for i in range(8)]
        bucket_map = {'0':0, '1':0, '2':1, '3':1, '4':2, '5':2, '6':3, '7':3, '8':4, '9':4, 'a':5, 'b':5, 'c':6, 'd':6, 'e':7, 'f':7}
        img_src = "S3"
    else:
        img_src = "HOUZZ"

    with open(os.path.join(index_dir, list_file), 'r') as f1:
        for line in f1:
            ct += 1
            space_id = line.strip()
            with open(os.path.join(annotation_dir, "{}.txt".format(space_id)), 'r') as f2:
                for row in f2:
                    external_image_id = row.strip().split('\t')[4] 
                    local_file = os.path.join(images_dir, space_id + ".jpg")
                    if img_src == "S3":
                        try:
                            key_str = '/%s/%s/%s-0-9.jpg' % (external_image_id[:3], external_image_id[4:6], external_image_id)
                            bucket_id = bucket_map[external_image_id[0]]
                            k = Key(bucket_list[bucket_id])
                            k.key = key_str
                            k.get_contents_to_filename(local_file)
                            success_ct += 1
                            print "{}. {}".format(ct, space_id)
                        except Exception as inst:
                            print "can not download image from S3, ", local_file, inst
                            print "continue x3 at line", (ct + 1)
                            fail_ct  += 1
                            fail_list.append((ct, space_id))
                    elif img_src == "HOUZZ":
                        try:
                            image_url = "http://st.houzz.com/simgs/" + external_image_id + "_9-0000/"
                            response = requests.get(image_url)
                            #print image_url, local_file
                            if response.status_code == 200:
                                with open(local_file, 'wb') as f:
                                    f.write(response.content)
                            print "{}. {}".format(ct, space_id)
                            success_ct += 1
                        except Exception as inst:
                            print "can not download image from url, " , local_file, inst
                            print "continue x4 at line", (ct + 1)
                            fail_ct  += 1
                            fail_list.append((ct, space_id))
                    else:
                        print "Did not specify a valid image source"
                        exit(1)
                    break
    
    print "Downloading completed."
    print "{} failed images:".format(len(fail_list))
    for ind, space_id in fail_list:
        print "{}. {}    ".format(ind, space_id),





def download_images_simple(list_file, images_dir):
    print "Start to download images according to this list: ", list_file
    ct = 0
    success_ct = 0
    fail_ct = 0
    fail_list = []
    img_src = ""
    hostname =  socket.gethostname()
    if hostname[:3] == "ip-":
        #aws ec2 instance
        from boto.s3.key import Key
        conn = boto.connect_s3()
        bucket_list = [conn.get_bucket("st00%d.houzz.com" % i) for i in range(8)]
        bucket_map = {'0':0, '1':0, '2':1, '3':1, '4':2, '5':2, '6':3, '7':3, '8':4, '9':4, 'a':5, 'b':5, 'c':6, 'd':6, 'e':7, 'f':7}
        img_src = "S3"
    else:
        img_src = "HOUZZ"

    with open(list_file, 'r') as f1:
        for line in f1:
            ct += 1
            space_id, external_image_id = line.strip().split()
            #with open(os.path.join(annotation_dir, "{}.txt".format(space_id)), 'r') as f2:
                #for row in f2:
                    #external_image_id = row.strip().split('\t')[4] 
            local_file = os.path.join(images_dir, space_id + ".jpg")
            if img_src == "S3":
                try:
                    key_str = '/%s/%s/%s-0-9.jpg' % (external_image_id[:3], external_image_id[4:6], external_image_id)
                    bucket_id = bucket_map[external_image_id[0]]
                    k = Key(bucket_list[bucket_id])
                    k.key = key_str
                    k.get_contents_to_filename(local_file)
                    success_ct += 1
                    print "{}. {}".format(ct, space_id)
                except Exception as inst:
                    print "can not download image from S3, ", local_file, inst
                    print "continue x3 at line", (ct + 1)
                    fail_ct  += 1
                    fail_list.append((ct, space_id))
            elif img_src == "HOUZZ":
                try:
                    image_url = "http://st.houzz.com/simgs/" + external_image_id + "_9-0000/"
                    response = requests.get(image_url)
                    #print image_url, local_file
                    if response.status_code == 200:
                        with open(local_file, 'wb') as f:
                            f.write(response.content)
                    print "{}. {}".format(ct, space_id)
                    success_ct += 1
                except Exception as inst:
                    print "can not download image from url, " , local_file, inst
                    print "continue x4 at line", (ct + 1)
                    fail_ct  += 1
                    fail_list.append((ct, space_id))
            else:
                print "Did not specify a valid image source"
                exit(1)
    
    print "Downloading completed."
    print "{} failed images:".format(len(fail_list))
    for ind, space_id in fail_list:
        print "{}. {}    ".format(ind, space_id),



def download_temp_images():
    img_list = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/temp.txt"
    img_dir = "./data/HouzzDataCollection/HouzzData1/Images_temp"
    download_images_simple(img_list, img_dir) 




def show_classes():
    cate_table_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_table_merged_modified.txt"
    all_classes = ['__background__']
    with open(cate_table_file, 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            all_classes.append(vals[-1])
    print all_classes





def check_purge_map_draft():
    #check 1: either # and ~
    purge_draft_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_purge_map_draft"
    with open(purge_draft_file, 'r') as fh:
        ct = 0
        for line in fh:
            ct += 1
            if '~' in line and '#' in line:
                print ct, '  ', line
        print "Check 1 completed"
    print '\n'


    #check 2: target-source relation
    target_cates_and_cts = {}
    contained_map = {}
    ct3 = 0
    del_cate_ct = 0
    mapped_cate_ct = 0
    del_box_ct = 0
    kept_box_ct = 0
    with open(purge_draft_file, 'r') as fh:
        for line in fh:
            ct3 += 1
            s = line.split()[1]
            box_ct = int( s[s.find('(') + 1 : s.find(')')] )
            if line[0] == '#':
                del_box_ct += box_ct
                del_cate_ct += 1
                continue
            mapped_cate_ct += 1
            kept_box_ct += box_ct
            vals = line.strip().split('\t')
            vals = filter(None, vals) #filter out empty strings caused by double tabs
            source_cate = vals[2]
            target_cate = None
            if '~' in line:
                target_cate = vals[3].split('~')[1]
            else:
                target_cate = vals[2]
            assert target_cate != '', "Target category is empty! Check line \n{}: {}".format(ct3, line)
            if target_cate in target_cates_and_cts:
                target_cates_and_cts[target_cate] += box_ct
            else:
                target_cates_and_cts[target_cate] = box_ct
            if target_cate in contained_map:
                contained_map[target_cate].append(source_cate)
            else:
                contained_map[target_cate] = [source_cate]
    sorted_cates_and_cts = sorted(target_cates_and_cts.items(), key = lambda x: x[1], reverse = True)
    contained_map_list = sorted(contained_map.items(),  key = lambda x: x[0])
    ct4 = 0
    for cate, ct in sorted_cates_and_cts:
        ct4 += 1
        assert cate in contained_map[cate], "target category '{}' not in its source categories!".format(cate)
        print "{}. {} ({}) : \t\t".format(ct4, cate, target_cates_and_cts[cate]),
        for item in contained_map[cate]:
            print '{},\t'.format(item),
        print '\n'

    print "del_cate_ct = ", del_cate_ct
    print "mapped_cate_ct = ", mapped_cate_ct
    print "total category ct = ", del_cate_ct + mapped_cate_ct
    print "Check 2 completed"




def gen_purge_map():
    #generate purge map
    purge_draft_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_purge_map_draft"
    cate_names_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/categories.txt"

    purge_map_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_purge_map.txt"
    cate_table_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_table_purged.txt"

    #categories that are not to be mapped or used in training
    skip_cates = ['paint and wall covering supplies',
                  'primers',
                  'stains and varnishes',
                  'paint']

    cate_name2id = {}
    ct = 0

    with open(cate_names_file, 'r') as fh:
        for line in fh:
            ct += 1
            vals = line.strip().split('\t')
            cid = vals[0]
            cate = vals[2]
            cate_name2id[cate] = cid 

    purge_map = {}
    ct = 0
    with open(purge_draft_file) as fh:
        for line in fh:
            ct += 1
            target_cate = None
            vals = line.strip().split('\t')
            vals = filter(None, vals) #filter out empty strings using None as function
            source_cate = vals[2]
            if line[0] == '#':
                #skipping paint related categories
                if source_cate in skip_cates:
                    continue
                target_cate = "products" #map every ignored category to products
            else:
                if '~' in line:
                    target_cate = vals[3].split('~')[1]
                else:
                    target_cate = vals[2]
            sid = cate_name2id[source_cate]
            tid = cate_name2id[target_cate]
            if sid in purge_map:
                raise Exception("Key in dict, surprisingly!")
            else:
                purge_map[sid] = tid

    sorted_map_list = sorted(purge_map.items(), key = lambda x: x[1])
    with open(purge_map_file, 'w') as fh:
        for sid, tid in sorted_map_list:
            fh.write("{} {}\n".format(sid, tid))
    #write purged category table for classfication
    #format: index, category_id, category_name 
    purged_cids =  sorted( list(set(purge_map.values())), key = lambda x: int(x) )
    cate_id2name = get_cate_name_map(cate_names_file)
    ct2 = 0
    with open(cate_table_file, 'w') as fh:
        for cid in purged_cids:
            ct2 += 1 #class index starting from 1, without background as 0
            fh.write("{}\t{}\t{}\n".format(ct2, cid, cate_id2name[cid]))



def load_cate_purge_map(filepath):
    return load_cate_merge_map(filepath)


def purge(cate_purge_map, orgl_file, dst_file):
    with open(dst_file, 'w') as fw:
        with open(orgl_file, 'r') as fr:
            for line in fr:
                vals = line.split('\t')
                if vals[-3] in cate_purge_map:
                    vals[-3] = cate_purge_map[vals[-3]]
                    fw.write( '\t'.join(vals))
                else:
                    #vals[-3] = '-1' #categories to be deleted
                    #categories not on purge map are ignored
                    pass

        
             

def purge_cates():
    cate_purge_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_purge_map.txt"

    ref_all_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015.txt"
    ref_train_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039.txt"
    ref_val_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_holdout2000.txt"
    ref_test_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_testset5000.txt"

    dst_all_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_cate_purged.txt"
    dst_train_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039_cate_purged.txt"
    dst_val_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_holdout2000_cate_purged.txt"
    dst_test_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_testset5000_cate_purged.txt"

    cate_purge_map = load_cate_purge_map(cate_purge_file)

    purge(cate_purge_map, ref_train_file, dst_train_file)
    purge(cate_purge_map, ref_val_file, dst_val_file)
    purge(cate_purge_map, ref_test_file, dst_test_file)
    purge(cate_purge_map, ref_all_file, dst_all_file)


def purge_cates_for_new_data():
    cate_purge_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/cat_purge_map.txt"

    #ref_all_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01122016.txt"
    #ref_all_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01272016.txt"
    ref_all_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_02052016.txt"

    #dst_all_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01122016_cate_purged.txt"
    #dst_all_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01272016_cate_purged.txt"
    dst_all_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_02052016_cate_purged.txt"

    cate_purge_map = load_cate_purge_map(cate_purge_file)

    purge(cate_purge_map, ref_all_file, dst_all_file)




def partition_purged_sets():
    print "Partition the purged boxes  into train, validation, and test, without overlapping space_ids"
    ref_train_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039_cate_purged.txt"
    ref_val_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_holdout2000_cate_purged.txt"
    ref_test_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_testset5000_cate_purged.txt"
    #test_space_num_to_set = 4800
    #val_space_num_to_set = 1800
    dest_path = "./data/HouzzDataCollection/HouzzData0/imageListsAll/cate_purged/"
    train_index_file = "train.txt"
    val_index_file = "val.txt"
    test_index_file = "test.txt"

    partition_sets(ref_train_file, ref_val_file, ref_test_file, dest_path, train_index_file, val_index_file, test_index_file)




def split_box_list_of_purged_cates(purge_cate = True):
    all_box_list = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015.txt"
    dest_dir = "./data/HouzzDataCollection/HouzzData0/annotationsAll/original/" 
    if purge_cate:
        all_box_list = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_cate_purged.txt"
        dest_dir = "./data/HouzzDataCollection/HouzzData0/annotationsAll/cate_purged/"
    print "all boxes: {}".format(all_box_list)
    print "dest dir: {}".format(dest_dir)
    split_box_list_by_space(all_box_list, dest_dir)
   



def split_box_list_of_purged_cates_for_new_data(purge_cate = True):
    #all_box_list = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01122016.txt"
    #all_box_list = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01272016.txt"
    all_box_list = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_02052016.txt"
    dest_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/original/" 
    if purge_cate:
        #all_box_list = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01122016_cate_purged.txt"
        all_box_list = all_box_list[:-4] + "_cate_purged.txt"
        dest_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged/"
    print "all boxes: {}".format(all_box_list)
    print "dest dir: {}".format(dest_dir)
    split_box_list_by_space(all_box_list, dest_dir)





def gen_image_list():
    anno_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged"
    #image_list_file = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/all_01122016.txt"
    #image_list_file = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/all_01272016.txt"
    image_list_file = "./data/HouzzDataCollection/HouzzData1/imageListsAll/imageLists_cate_purged/all_02052016.txt"

    img_list = glob.glob(os.path.join(anno_dir, "*.txt"))
    #filter by date
    #target_date = "Jan 12"
    #target_date = "Jan 27"
    target_date = "Feb  5"
    img_list = [ f for f in img_list if target_date in time.ctime(os.path.getctime(f)) ]
    #print "img_list"
    #embed()

    img_list = [ os.path.basename(item)[:-4] for item in img_list]
    #print "img_list barename"
    #embed()
    output_content = '\n'.join(img_list)

    with open(image_list_file, 'w') as fw:
        fw.write(output_content)





def create_small_set():
    import shutil
    train_list_file = "./data/HouzzDataCollection1/HouzzData0/ImageLists/train3000.txt"
    img_src_path = "./data/HouzzDataCollection/HouzzData0/Images/"
    img_dst_path = "./data/HouzzDataCollection1/HouzzData0/Images/"
    anno_src_path = "./data/HouzzDataCollection/HouzzData0/Annotations/"
    anno_dst_path = "./data/HouzzDataCollection1/HouzzData0/Annotations/"
    with open(train_list_file, 'r') as fh:
        ct = 0
        for line in fh:
            ct += 1
            barename = line.strip()
            print ct, '  ', barename
            shutil.copy2( os.path.join(img_src_path, barename+'.jpg'),
                            os.path.join(img_dst_path, barename+'.jpg'))
            shutil.copy2( os.path.join(anno_src_path, barename+'.txt'),
                            os.path.join(anno_dst_path, barename+'.txt'))



def get_cate2cls_map(cate2cls_file):
    cate2cls = {}
    with open(cate2cls_file, 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            cls_ind = int(vals[0])
            cate_id = vals[1]
            cate2cls[cate_id] = cls_ind
    return cate2cls




def convert_gt_to_det_format():

    gtruth_dir = "./data/HouzzDataCollection/HouzzData0/Annotations"
    test_img_list = "./data/HouzzDataCollection/HouzzData0/ImageLists/val.txt"
    dst_dir = "./work_data/faster-rcnn/VGG16/dets_i80000_train40000/pseudo_rst"
    cate2cls_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_table_purged.txt"

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir) 
    cate2cls = get_cate2cls_map(cate2cls_file) 
    ct = 0
    file_list = [] 
    with open(test_img_list, 'r') as fr:
        for line in fr:
            file_list.append(line.strip() + '.txt')

    for basename in file_list:
        ct += 1
        print ct, ' ', basename
        output_list = []
        file_pathname = os.path.join(gtruth_dir, basename)
        with open(file_pathname, 'r') as fr:
            for line in fr:
                vals = line.split('\t')
                cate_id = vals[-3]
                cls_ind = cate2cls[cate_id]
                x, y, w, h = vals[7:11]
                ###
                #w = str(int(w)/2) #change width to test the overlap ratio thresh
                ###
                score = 1.0
                output_list.append( "{} {} {} {} {} {}\n".format(\
                        x, y, w, h, cls_ind, score) )
       
        output_content = ''.join(output_list)
        dst_pathname = os.path.join(dst_dir, basename)
        with open(dst_pathname, 'w') as fw:
            fw.write(output_content)
            ###add extra boxes to verify the precision
            fw.write(output_content)
            ###



def plot_training_error_curve():
    #plot training error curve for alt opt training
    import matplotlib.pyplot as plt

    #logfile = "/Users/houzz/Work/work_data/faster-rcnn/py-faster-rcnn/experiments/logs/faster_rcnn_alt_opt_VGG16_.txt.2015-12-28_03-07-35"
    #logfile = "/Users/houzz/Work/work_data/faster-rcnn/py-faster-rcnn/experiments/logs/faster_rcnn_alt_opt_VGG16_.txt.2015-12-18_22-08-01"
    #logfile = "/Users/houzz/Work/work_data/faster-rcnn/py-faster-rcnn/experiments/logs/faster_rcnn_alt_opt_g1_ZF_.txt.2015-12-28_03-32-54"

    #logfile = "./experiments/logs/faster_rcnn_alt_opt_VGG16_.txt.2015-12-18_22-08-01"
    #logfile = "./experiments/logs/faster_rcnn_alt_opt_VGG16_.txt.2015-12-28_03-07-35"
    #logfile = "./experiments/logs/faster_rcnn_alt_opt_g1_ZF_.txt.2015-12-28_03-32-54"
    #logfile = "./experiments/logs/faster_rcnn_alt_opt_g1_ZF_.txt.2016-01-07_02-17-32"
    #logfile = "./experiments/logs/faster_rcnn_alt_opt_ZF_.txt.2016-01-07_21-04-15"
    #logfile = "./experiments/logs/faster_rcnn_alt_opt_VGG16_.txt.2016-01-07_01-23-38"
    logfile = "./experiments/logs/faster_rcnn_alt_opt_houzzdata1_VGG16_.txt.2016-01-18_18-38-59"

    curve_figure_file = logfile + '.curve.pdf'
    ma_win_size = 100
    downsample_rate = 10
    #Wrong preallocation: loss_bbox_list[0] is loss_bbox_list[1]
    #loss_bbox_list = [[None] * 299000] * 2 
    #loss_cls_list = [[None] * 299000] * 2 
    #loss_rpn_bbox_list = [[None] * 299000] * 2 
    #loss_rpn_cls_list = [[None] * 299000] * 2 
    #Correct preallocation
    a = [None] * 299000
    loss_bbox_list = [a[:], a[:]] 
    loss_cls_list = [a[:], a[:]] 
    loss_rpn_bbox_list = [a[:], a[:]] 
    loss_rpn_cls_list = [a[:], a[:]] 
    lr_list = [[], []] 
    rpn_lr_list = [[], []]
    itr_list = [a[:], a[:]]
    rpn_itr_list = [a[:], a[:]]

    p_bbox = re.compile(r"(loss|bbox)_(loss|bbox) = ([\d.]+)")
    p_cls = re.compile(r"(loss|cls)_(loss|cls) = ([\d.]+)")
    p_rpn_bbox = re.compile(r"rpn_(loss|bbox)_(loss|bbox) = ([\d.]+)")
    p_rpn_cls = re.compile(r"rpn_(loss|cls)_(loss|cls) = ([\d.]+)")
    p_itr_lr = re.compile(r"Iteration ([\d]+), lr = ([\d.e-]+)")
    ct_bbox = [0] * 2
    ct_cls = [0] * 2
    ct_rpn_bbox = [0] * 2
    ct_rpn_cls = [0] * 2
    ct_itr = [0] * 2
    ct_rpn_itr = [0] * 2 
    #start_marker = "Stage 2 Fast R-CNN"
    #end_marker = "aaaaaaaaaaaaaaaa"
    #start_marker = "Stage 1 Fast R-CNN"
    #end_marker = "Stage 2 RPN, init"
    marker0 = "Stage 1 RPN, init"
    marker1 = "Stage 1 Fast R-CNN"
    marker2 = "Stage 2 RPN, init"
    marker3 = "Stage 2 Fast R-CNN"
    section = None
    with open(logfile, 'r') as fr:
        for line in fr:

            if marker0 in line:
                section = 0
            if marker1 in line:
                section = 1
            if marker2 in line:
                section = 2
            if marker3 in line:
                section = 3
            if section is None:
                continue
            stage = section / 2 
            if section == 0 or section == 2: #rpn training
                if "Train net output #0: rpn_" in line:
                    matched_bbox = p_rpn_cls.findall(line)
                    assert len(matched_bbox) == 1, "Matched locations is not one!"
                    loss_rpn_cls_list[stage][ct_rpn_cls[stage]] = float(matched_bbox[0][2])
                    ct_rpn_cls[stage] += 1
                elif "Train net output #1: rpn_" in line: 
                    matched_cls = p_rpn_bbox.findall(line)
                    assert len(matched_cls) == 1, "Matched locations is not one!"
                    loss_rpn_bbox_list[stage][ct_rpn_bbox[stage]] = float(matched_cls[0][2])
                    ct_rpn_bbox[stage] += 1
                elif "lr = " in line:
                    matched = p_itr_lr.findall(line)
                    assert len(matched) == 1, "Matched locations is not one!"
                    itr, lr = int(matched[0][0]), float(matched[0][1])
                    rpn_itr_list[stage][ct_rpn_itr[stage]] = itr
                    if rpn_lr_list[stage] == [] or rpn_lr_list[stage][-1][1] != lr:
                        rpn_lr_list[stage].append((ct_rpn_itr[stage], lr))
                    ct_rpn_itr[stage] += 1
                else:
                    continue
                if ct_rpn_bbox[stage] % 1000 == 0:
                    print "stage {}, ct_rpn_bbox = {},  ct_rpn_cls = {}".format(stage + 1, ct_rpn_bbox[stage], ct_rpn_cls[stage])
            elif section == 1 or section == 3: #fast r-cnn training
                if "Train net output #0" in line:
                    matched_bbox = p_bbox.findall(line)
                    assert len(matched_bbox) == 1, "Matched locations is not one!"
                    loss_bbox_list[stage][ct_bbox[stage]] = float(matched_bbox[0][2])
                    ct_bbox[stage] += 1
                elif "Train net output #1" in line: 
                    matched_cls = p_cls.findall(line)
                    assert len(matched_cls) == 1, "Matched locations is not one!"
                    loss_cls_list[stage][ct_cls[stage]] = float(matched_cls[0][2])
                    ct_cls[stage] += 1
                elif "lr = " in line:
                    matched = p_itr_lr.findall(line)
                    assert len(matched) == 1, "Matched locations is not one!"
                    itr, lr = int(matched[0][0]), float(matched[0][1])
                    itr_list[stage][ct_itr[stage]] = itr
                    if lr_list[stage] == [] or lr_list[stage][-1][1] != lr:
                        lr_list[stage].append((ct_itr[stage], lr))
                    ct_itr[stage] += 1
                else:
                    continue
                if ct_cls[stage] % 1000 == 0:
                    print "stage {}, ct_bbox = {},  ct_cls = {}".format(stage + 1, ct_bbox[stage], ct_cls[stage])
    
    #assert ct_bbox == ct_cls, "Number of bbox and cls losses not equal!"
    #print "middle check"
    #embed()

    plt.figure(1) 
    plt.suptitle('Moving average window = {}'.format(ma_win_size))
    for stage in range(2):
        #plt.subplot(4, 2, 1 + stage)
        plt.subplot(2, 2, 1 + stage)
        plt.tight_layout()
        #plt.title("rpn bbox loss")
        plt.title("stage {}".format(stage+1))
        if ct_rpn_bbox[stage] > 0:
            loss_rpn_bbox_list[stage] = loss_rpn_bbox_list[stage][:ct_rpn_bbox[stage]]
            loss_rpn_bbox_list_ma = movingaverage(np.array(loss_rpn_bbox_list[stage]), ma_win_size)
            loss_rpn_bbox_list_ma = loss_rpn_bbox_list_ma[::downsample_rate]
            plt.plot(loss_rpn_bbox_list_ma, 'r.', label = "rpn bbox loss")
            plt.ylim([-0.01, 0.2])
            plt.legend(loc = "upper right", fontsize = 8)

        #plt.subplot(4, 2, 3 + stage)
        #plt.tight_layout()
        #plt.title("rpn classification loss")
        if ct_rpn_cls[stage] > 0:
            loss_rpn_cls_list[stage] = loss_rpn_cls_list[stage][:ct_rpn_cls[stage]]
            loss_rpn_cls_list_ma = movingaverage(np.array(loss_rpn_cls_list[stage]), ma_win_size)
            loss_rpn_cls_list_ma = loss_rpn_cls_list_ma[::downsample_rate]
            plt.plot(loss_rpn_cls_list_ma, 'b.', label = "rpn cls loss")
            plt.ylim([-0.01, 0.2])
            #plt.ylim([-0.01, 0.2])
            plt.legend(loc = "upper right", fontsize = 8)

        #plt.subplot(4, 2, 5 + stage) 
        plt.subplot(2, 2, 3 + stage) 
        plt.tight_layout() #increase the space between the plots
        #plt.title("bbox loss")
        plt.title("stage {}".format(stage+1))
        if ct_bbox[stage] > 0:
            loss_bbox_list[stage] = loss_bbox_list[stage][:ct_bbox[stage]]
            loss_cls_list[stage] = loss_cls_list[stage][:ct_cls[stage]]
            itr_list_temp, loss_bbox_list_temp, loss_cls_list_temp = smooth_curve(itr_list[stage], loss_bbox_list[stage], loss_cls_list[stage], ma_win_size)
            #loss_bbox_list_ma = movingaverage(np.array(loss_bbox_list[stage]), ma_win_size)
            itr_list_temp, loss_bbox_list_temp, loss_cls_list_temp, lr_list_temp = downsample_data(itr_list_temp, loss_bbox_list_temp, loss_cls_list_temp, lr_list[stage], downsample_rate)

            #loss_bbox_list_ma = loss_bbox_list_ma[::downsample_rate]
            #plt.plot(itr_list[stage][::downsample_rate], loss_bbox_list_ma, 'r.', label = "bbox loss")

            plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            plt.plot(itr_list_temp, loss_bbox_list_temp, 'r.', label = "bbox loss")
            plt.plot(itr_list_temp, loss_cls_list_temp, 'b.', label="cls loss")
            plt.tick_params(axis='both', which='major', labelsize=8)
            #plt.ylim([-0.01, 0.3])
            plt.ylim([-0.01, 1.0])
            for ind, lr in lr_list_temp:
                if ind >= len(itr_list_temp):
                    continue
                #annotate_point = (len(loss_bbox_list_ma)/2, loss_bbox_list_ma[len(loss_bbox_list_ma)/2])
                annotate_point = (itr_list_temp[ind], loss_bbox_list_temp[ind])
                plt.annotate('lr={}'.format(lr), xy=annotate_point, xytext=(annotate_point[0], annotate_point[1] - 0.2),\
                        arrowprops=dict(facecolor='black', shrink=0.05, width = 1, headwidth = 4), fontsize = 8)
            plt.legend(loc = "upper right", fontsize = 8)
        
        #plt.subplot(4, 2, 7 + stage)
        #plt.tight_layout()
        #plt.title("classification loss")

        #if ct_cls[stage] > 0:
            #loss_cls_list[stage] = loss_cls_list[stage][:ct_cls[stage]]
            #itr_list_temp, loss_cls_list_temp = smooth_curve(itr_list[stage], loss_cls_list[stage], ma_win_size)
            #itr_list_temp, loss_cls_list_temp, lr_list_temp = downsample_data(itr_list_temp, loss_cls_list_temp, lr_list[stage], downsample_rate)
            #loss_cls_list_ma = movingaverage(np.array(loss_cls_list[stage]), ma_win_size)
            #loss_cls_list_ma = loss_cls_list_ma[::downsample_rate]
            #plt.ylim([-0.01, 1.0])

    #print "final check"
    #embed()

    plt.savefig(curve_figure_file)




def plot_training_error_curve2():
    #plot training error curve for end2end training
    import matplotlib.pyplot as plt

    #logfile = "./experiments/logs/faster_rcnn_VGG16_.txt.2016-01-11_19-20-11"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_VGG16_.txt.2016-01-18_18-38-10"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_VGG16_--set_TRAIN.SNAPSHOT_INFIX_t0000.txt.2016-01-26_14-32-40"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_VGG16_--set_TRAIN.SNAPSHOT_INFIX_n0001.txt.2016-01-28_23-08-17"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_VGG16_--set_TRAIN.SNAPSHOT_INFIX_n0002.txt.2016-01-31_19-41-47"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_VGG16_--set_TRAIN.SNAPSHOT_INFIX_n0003.txt.2016-01-31_21-03-16"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_VGG16_--set_TRAIN.SNAPSHOT_INFIX_n0007.txt.2016-02-02_18-22-54"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_VGG16_--set_TRAIN.SNAPSHOT_INFIX_n0008.txt.2016-02-03_18-20-02"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_VGG16_--set_TRAIN.SNAPSHOT_INFIX_n0016.txt.2016-02-15_17-35-06"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_VGG16_--set_TRAIN.SNAPSHOT_INFIX_n0017.txt.2016-02-17_16-52-02"
    #logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_RESNET_--set_TRAIN.SNAPSHOT_INFIX_n0020.txt.2016-02-19_16-08-22"
    logfile = "./experiments/logs/faster_rcnn_end2end_houzzdata1_RESNET_--set_TRAIN.SNAPSHOT_INFIX_n0021.txt.2016-02-20_13-29-54"

    curve_figure_file = logfile + '.curve.pdf'
    ma_win_size = 100
    downsample_rate = 10
    #Wrong preallocation: loss_bbox_list[0] is loss_bbox_list[1]
    #loss_bbox_list = [[None] * 299000] * 2 
    #loss_cls_list = [[None] * 299000] * 2 
    #loss_rpn_bbox_list = [[None] * 299000] * 2 
    #loss_rpn_cls_list = [[None] * 299000] * 2 
    #Correct preallocation
    a = [None] * 299000
    loss_bbox_list = a[:] 
    loss_cls_list = a[:] 
    loss_rpn_bbox_list = a[:]
    loss_rpn_cls_list = a[:]
    itr_list = a[:]
    lr_list = [] 

    p_bbox = re.compile(r"(loss|bbox)_(loss|bbox) = ([\d.]+)")
    p_cls = re.compile(r"(loss|cls)_(loss|cls) = ([\d.]+)")
    p_rpn_bbox = re.compile(r"rpn_(loss|bbox)_(loss|bbox) = ([\d.]+)")
    p_rpn_cls = re.compile(r"rpn_(loss|cls)_(loss|cls) = ([\d.]+)")
    p_itr_lr = re.compile(r"Iteration ([\d]+), lr = ([\d.e-]+)")
    ct_bbox = 0
    ct_cls = 0
    ct_rpn_bbox = 0
    ct_rpn_cls = 0
    ct_itr = 0
    with open(logfile, 'r') as fr:
        for line in fr:
            if "Train net output #0" in line:
                matched = p_bbox.findall(line)
                assert len(matched) == 1, "Matched locations is not one!"
                loss_bbox_list[ct_bbox] = float(matched[0][2])
                ct_bbox += 1
            elif "Train net output #1" in line: 
                matched = p_cls.findall(line)
                assert len(matched) == 1, "Matched locations is not one!"
                loss_cls_list[ct_cls] = float(matched[0][2])
                ct_cls += 1
            elif "Train net output #2: rpn_" in line:
                matched = p_rpn_cls.findall(line)
                assert len(matched) == 1, "Matched locations is not one!"
                loss_rpn_cls_list[ct_rpn_cls] = float(matched[0][2])
                ct_rpn_cls += 1
            elif "Train net output #3: rpn_" in line: 
                matched = p_rpn_bbox.findall(line)
                assert len(matched) == 1, "Matched locations is not one!"
                loss_rpn_bbox_list[ct_rpn_bbox] = float(matched[0][2])
                ct_rpn_bbox += 1
                if ct_rpn_bbox % 500 == 0 and ct_rpn_bbox != 0:
                    print "ct_rpn_bbox = {},  ct_rpn_cls = {}".format( ct_rpn_bbox, ct_rpn_cls)
            elif "lr = " in line:
                matched = p_itr_lr.findall(line)
                assert len(matched) == 1, "Matched locations is not one!"
                itr, lr = int(matched[0][0]), float(matched[0][1])
                itr_list[ct_itr] = itr
                if lr_list == [] or lr_list[-1][1] != lr:
                    lr_list.append((ct_itr, lr))
                ct_itr += 1
            else:
                continue
    
    #assert ct_bbox == ct_cls, "Number of bbox and cls losses not equal!"
    #print "middle check"
    #embed()

    plt.figure(1) 
    plt.suptitle('Moving average window = {}, downsample rate = {}'.format(ma_win_size, downsample_rate))
    #for stage in range(2):
    #plt.subplot(4, 2, 1 + stage)
    plt.subplot(2, 1, 1)
    plt.tight_layout()
    #plt.title("rpn bbox loss")
    if ct_rpn_bbox > 0:
        #loss_rpn_bbox_list = loss_rpn_bbox_list[:ct_rpn_bbox]
        #loss_rpn_bbox_list_ma = movingaverage(np.array(loss_rpn_bbox_list), ma_win_size)
        #loss_rpn_bbox_list_ma = loss_rpn_bbox_list_ma[::downsample_rate]
        #plt.plot(loss_rpn_bbox_list_ma, 'r.', label = "rpn bbox loss")
        #plt.ylim([-0.01, 0.2])
        #plt.legend(loc = "upper right", fontsize = 8)

        
        loss_rpn_bbox_list = loss_rpn_bbox_list[:ct_rpn_bbox]
        loss_rpn_cls_list = loss_rpn_cls_list[:ct_rpn_cls]
        itr_list_temp, loss_rpn_bbox_list_temp, loss_rpn_cls_list_temp = smooth_curve(itr_list, loss_rpn_bbox_list, loss_rpn_cls_list, ma_win_size)
        itr_list_temp, loss_rpn_bbox_list_temp, loss_rpn_cls_list_temp, lr_list_temp = downsample_data(itr_list_temp, loss_rpn_bbox_list_temp, loss_rpn_cls_list_temp, lr_list, downsample_rate)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.plot(itr_list_temp, loss_rpn_bbox_list_temp, 'r.', label = "rpn bbox loss")
        plt.plot(itr_list_temp, loss_rpn_cls_list_temp, 'b.', label="rpn cls loss")
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.ylim([-0.01, 0.2])
        for ind, lr in lr_list_temp:
            if ind >= len(itr_list_temp):
                continue
            #annotate_point = (len(loss_bbox_list_ma)/2, loss_bbox_list_ma[len(loss_bbox_list_ma)/2])
            annotate_point = (itr_list_temp[ind], loss_rpn_bbox_list_temp[ind])
            plt.annotate('lr={}'.format(lr), xy=annotate_point, xytext=(annotate_point[0], annotate_point[1] + 0.04),\
                    arrowprops=dict(facecolor='black', shrink=0.05, width = 1, headwidth = 4), fontsize = 8)
        plt.legend(loc = "upper right", fontsize = 8)

    #if ct_rpn_cls > 0:
        #loss_rpn_cls_list = loss_rpn_cls_list[:ct_rpn_cls]
        #loss_rpn_cls_list_ma = movingaverage(np.array(loss_rpn_cls_list), ma_win_size)
        #loss_rpn_cls_list_ma = loss_rpn_cls_list_ma[::downsample_rate]
        #plt.plot(loss_rpn_cls_list_ma, 'b.', label = "rpn cls loss")
        #plt.ylim([-0.01, 0.2])
        ##plt.ylim([-0.01, 0.2])
        #plt.legend(loc = "upper right", fontsize = 8)

    #plt.subplot(4, 2, 5 + stage) 
    plt.subplot(2, 1, 2) 
    plt.tight_layout() #increase the space between the plots
    if ct_bbox > 0:
        loss_bbox_list = loss_bbox_list[:ct_bbox]
        loss_cls_list = loss_cls_list[:ct_cls]
        itr_list_temp, loss_bbox_list_temp, loss_cls_list_temp = smooth_curve(itr_list, loss_bbox_list, loss_cls_list, ma_win_size)
        #loss_bbox_list_ma = movingaverage(np.array(loss_bbox_list), ma_win_size)
        itr_list_temp, loss_bbox_list_temp, loss_cls_list_temp, lr_list_temp = downsample_data(itr_list_temp, loss_bbox_list_temp, loss_cls_list_temp, lr_list, downsample_rate)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.plot(itr_list_temp, loss_bbox_list_temp, 'r.', label = "bbox loss")
        plt.plot(itr_list_temp, loss_cls_list_temp, 'b.', label="cls loss")
        plt.tick_params(axis='both', which='major', labelsize=8)
        #plt.ylim([-0.01, 0.3])
        plt.ylim([-0.01, 1.0])
        for ind, lr in lr_list_temp:
            if ind >= len(itr_list_temp):
                continue
            #annotate_point = (len(loss_bbox_list_ma)/2, loss_bbox_list_ma[len(loss_bbox_list_ma)/2])
            annotate_point = (itr_list_temp[ind], loss_bbox_list_temp[ind])
            plt.annotate('lr={}'.format(lr), xy=annotate_point, xytext=(annotate_point[0], annotate_point[1] + 0.2),\
                    arrowprops=dict(facecolor='black', shrink=0.05, width = 1, headwidth = 4), fontsize = 8)
        plt.legend(loc = "upper right", fontsize = 8)

    #print "final check"
    #embed()

    plt.savefig(curve_figure_file)



def smooth_curve(x, y1, y2, win_size):
    new_y1 = movingaverage(np.array(y1), win_size)
    new_y2 = movingaverage(np.array(y2), win_size)
    len_y1 = len(new_y1)
    len_y2 = len(new_y2)
    assert len_y1 == len_y2, "Lengths differ!"
    new_x = x[:len_y1]
    return new_x, new_y1, new_y2


def downsample_data(x, y1, y2, lr_list, downsample_rate):
    new_x = x
    new_y1 = y1
    new_y2 = y2
    new_lr_list = lr_list
    return new_x, new_y1, new_y2, new_lr_list




def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    #print "movingaverage"
    #embed()
    #return np.convolve(interval, window, 'same')
    return np.convolve(interval, window, 'valid')




def draw_training_gt_boxes():
    dataset_name = "houzzdata1"
    img_list_file = "./data/HouzzDataCollection/HouzzData1/ImageLists/train100.txt"
    #img_list_file = "./data/HouzzDataCollection/HouzzData0/ImageLists/train110.txt"
    img_dir = "./data/HouzzDataCollection/HouzzData1/Images"
    anno_dir = "./data/HouzzDataCollection/HouzzData1/Annotations"
    #dest_path = "./work_data/faster-rcnn/houzzdata0_train_vis/"
    dest_path = "./work_data/faster-rcnn/train_vis/"
    with open(img_list_file, 'r') as fr:
        all_content = fr.read()

    img_list = all_content.split('\n')

    for barename in img_list:
        if barename == '':
            continue
        img_name = os.path.join(img_dir, barename + ".jpg")
        anno_name = os.path.join(anno_dir, barename + ".txt")
        img = cv2.imread(img_name)      
        thumb_img_size = PIL.Image.open(img_name).size
        thumb_img_wid = thumb_img_size[0]
        thumb_img_hei = thumb_img_size[1] 
        scale = 1.0
        boxes = []
        with open(anno_name, 'r') as fh:
            for line in fh:
                vals = line.strip().split('\t')
                orgl_img_wid = float(vals[5])
                scale = thumb_img_wid / orgl_img_wid
                wid = float(vals[9])
                hei = float(vals[10])
                if dataset_name == "houzzdata0":
                    x1 = float(vals[7])
                    y1 = float(vals[8])
                if dataset_name == "houzzdata1":
                    xc = float(vals[7]) #center x
                    yc = float(vals[8]) #center y
                    x1 = xc - wid / 2 
                    y1 = yc - hei / 2
                x2 = x1 + wid - 1
                y2 = y1 + hei - 1
                x1 = np.clip(x1 * scale, 0.0, thumb_img_wid - 1)
                x2 = np.clip(x2 * scale, 0.0, thumb_img_wid - 1)
                y1 = np.clip(y1 * scale, 0.0, thumb_img_hei - 1)
                y2 = np.clip(y2 * scale, 0.0, thumb_img_hei - 1)
                #categoryID = int(vals[-3])
                #if categoryID not in self._categoryID_to_ind:
                    #print "Likely keyvalue error."
                    #print "vals: ", vals
                    #categoryID = 2002 # wrong keys mapped to "tabel top"
                #cls = self._categoryID_to_ind[categoryID]
                #boxes[ct, :] = [x1, y1, x2, y2]
                #gt_classes[ct] = cls
                #overlaps[ct, cls] = 1.0
                #ct += 1
                boxes.append( [x1, y1, x2, y2])
        class_inds = [1] * len(boxes)
        vis_detections(dest_path, barename + ".jpg", img, class_inds, np.array(boxes))




def vis_detections(dest_path, image_basename, im, class_inds, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = range(dets.shape[0])
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    ct = 0
    for i in inds:
        #if class_inds[i] in GM.banned_categories:
            #continue
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
        #ax.text(bbox[0], bbox[1] - 2,
                #'{:d}. {:s} {:.3f}'.format(ct, CLASSES[class_inds[i]], score),
                #bbox=dict(facecolor='blue', alpha=0.5),
                #fontsize=14, color='white')
        #GM.detection_ct += 1

    ax.set_title(('{} detections with '
                 'confidence >= {:.2f}').format(ct, thresh), fontsize=14)
    print('{} detections'.format(ct))

    plt.axis('off')
    plt.tight_layout()
    #plt.draw()
    plt.savefig(os.path.join(dest_path, image_basename ))
    plt.close()




def check_image_lists_overlap():
    src0 = "./data/HouzzDataCollection/HouzzData1/ImageLists/train.txt"
    #src1 = "./data/HouzzDataCollection/HouzzData1/ImageLists/val.txt"
    src1 = "./data/HouzzDataCollection/HouzzData1/ImageLists/test.txt"
    train_set  = set()
    overlap_list = []
    with open(src0, 'r') as fr0:
        for line in fr0:
            train_set.add(int(line.strip()))
    with open(src1, 'r') as fr1:
        for line in fr1:
            space_id = int(line.strip())
            if space_id in train_set:
                overlap_list.append(space_id)
    print "Overlap list:", overlap_list





def get_bootstrapping_gt_from_final_prediction():
    from itertools import izip
    #Image list to generate bootstrapping bboxes on
    target_img_list_file = "./data/HouzzDataCollection/HouzzData1/ImageLists/train34000+6000.txt"
    #Original box file used to construct map from external id to space id
    target_box_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/image_tag_bounds_01122016+01272016.txt"
    #Files that contain detection+recognition results on 7 million photos
    pred_bbox_file_pattern = "/drive3/bbox/bboxes_*_*_classify"

    with open(target_img_list_file, 'r') as fr0:
        target_img_list = fr0.read().split()

    target_tot = len(target_img_list)
    target_img_list_iter = iter(target_img_list)
    zeros_list_iter = iter([0] * target_tot)
    target_img_btsp_box_count = dict( izip(target_img_list_iter, zeros_list_iter) )
    print "A total of {} target images".format(len(target_img_btsp_box_count)) 
    img_ext_id_to_space_id = dict()

    with open(target_box_file, 'r') as fr1:
        for line in fr1:
            vals = line.strip().split('\t')
            space_id = vals[1]
            ext_id = vals[4]
            if space_id in target_img_btsp_box_count:
                img_ext_id_to_space_id[ext_id] = space_id

    print "A total of {} external ids added".format(len(img_ext_id_to_space_id.keys()))

    print "Bounding box files:"
    bbox_file_list = glob.glob(pred_bbox_file_pattern)
    line_prefix = "/drive3/space_images/2/space_"
    line_prefix_len = len(line_prefix)
    file_ct = 0
    for bbox_file in bbox_file_list:
        file_ct += 1
        print file_ct, "  ", bbox_file
        line_ct = 0
        with open(bbox_file, 'r') as fr2:
            print "line count: ",
            for line in fr2:
                line_ct += 1
                if line_ct % 1000000 == 0:
                    print " ", line_ct,
                    sys.stdout.flush()
                if "count" in line:
                    continue
                vals = line.split()
                ext_id = vals[0][line_prefix_len:-4]
                if ext_id not in img_ext_id_to_space_id:
                    continue
                space_id = img_ext_id_to_space_id[ext_id]
                target_img_btsp_box_count[space_id] += 1

        print "\n"
    
    sorted_list = sorted(target_img_btsp_box_count.items(), key=lambda x: x[1], reverse = True) 
    rst_file = target_img_list_file + ".btsp.txt"
    with open(rst_file, 'w') as fw0:
        for item in sorted_list:
            fw0.write("{}\t{}\n".format(item[0], item[1]))
    

            


def get_bootstrapping_gt_from_detection():
    det_dir = "./work_data/faster-rcnn/VGG16/dets_i136040_train68000_houzzdata1_train34000+6000/t0.1" 
    orig_anno_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged"
    img_dir = "./data/HouzzDataCollection/HouzzData1/Images"
    cls_cate_map_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/cat_table_purged.txt"
    #rst_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged_btsp1"
    rst_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged_btsp2"

    boxnum_thresh = 6    
    #score_thresh = 0.5
    score_thresh = 0.7
    
    distb_before = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    distb_after = distb_before.copy()
    tot_box_before = 0
    tot_box_after = 0
    img_ct = 0

    cls2cate = dict() 
    with open(cls_cate_map_file, 'r') as fr:
        for line in fr:
            vals = line.strip().split('\t')
            cls2cate[int(vals[0])] = int(vals[1])

    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir)
    anno_files = glob.glob( os.path.join(orig_anno_dir, "*.txt") )
    file_ct = 0
    for pathname in anno_files:
        file_ct += 1
        basename = os.path.basename(pathname)
        print file_ct, ' ', basename
        rst_file = os.path.join(rst_dir, basename)
        content = []
        with open(pathname, 'r') as fr0:
            content = fr0.readlines()
        assert content[0] != '\n' and content[0] != ''
        orig_box_num = len(content)
        det_file = os.path.join(det_dir, basename)
        if not os.path.exists(det_file):
            #not belong to training set
            shutil.copyfile(pathname, rst_file) 
            continue
        elif orig_box_num >= boxnum_thresh:
            shutil.copyfile(pathname, rst_file)
            distb_before[boxnum_thresh] += 1
            tot_box_before += len(content) 
            img_ct += 1
            continue
        else:
            distb_before[orig_box_num] += 1
            tot_box_before += len(content) 
            img_ct += 1
            vals = content[0].split('\t')
            orig_img_wid = float(vals[5])
            orig_img_hei = float(vals[6])
            img_name = os.path.join(img_dir, basename[:-4] + '.jpg') 
            thumb_img_size = PIL.Image.open(img_name).size
            thumb_img_wid = thumb_img_size[0]
            thumb_img_hei = thumb_img_size[1] 
            scale = thumb_img_wid / orig_img_wid
            added_ct = 0
            with open(det_file, 'r') as fr1:
                for line in fr1:
                    if orig_box_num + added_ct >= boxnum_thresh:
                        break
                    assert line != '\n' and line != ''
                    vs = line.strip().split()
                    score = float(vs[5])
                    #Detection boxes are ordered by scores in descending order
                    if score < score_thresh:
                        break
                    wid = float(vs[2])
                    hei = float(vs[3])
                    xc = float(vs[0]) + wid / 2 #center
                    yc = float(vs[1]) + hei / 2 #center
                    xc = int(np.clip(xc / scale, 0.0, orig_img_wid - 1))
                    yc = int(np.clip(yc / scale, 0.0, orig_img_hei - 1))
                    wid = int(np.clip(wid / scale, 0.0, orig_img_wid))
                    hei = int(np.clip(hei / scale, 0.0, orig_img_hei))
                    cate = cls2cate[int(vs[4])]
                    vals[7] = str(xc)
                    vals[8] = str(yc)
                    vals[9] = str(wid)
                    vals[10] = str(hei)
                    vals[11] = "target_space_id" #just place holder
                    vals[12] = str(cate)
                    vals[13] = "target_image_id"
                    vals[14] = "target_external_id"
                    line_to_add = '\t'.join(vals) + '\n' 
                    content.append(line_to_add)
                    added_ct += 1
            if len(content) >= boxnum_thresh:
                distb_after[boxnum_thresh] += 1
            else:
                distb_after[len(content)] += 1
            tot_box_after += len(content)
            with open(rst_file, 'w') as fw:
                fw.write(''.join(content))

    print "image count: ", img_ct
    print "distb_before: ", distb_before
    print "distb_after: ", distb_after
    print "average box before: {:.1f}".format( float(tot_box_before) / img_ct)
    print "average box after: {:.1f}".format(float(tot_box_after) / img_ct)




def bboxes_overlap(btsp_bbox, orig_bbox):
    b1 = [orig_bbox[0], orig_bbox[1], orig_bbox[0]+orig_bbox[2]-1, orig_bbox[1]+orig_bbox[3]-1]
    b2 = [btsp_bbox[0], btsp_bbox[1], btsp_bbox[0]+btsp_bbox[2]-1, btsp_bbox[1]+btsp_bbox[3]-1]
    bi = [max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])] 
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    ov = 0.0
    if iw > 0 and ih > 0:
        ua = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1) + \
            (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1) - \
            iw * ih
        ov = float(iw * ih) / ua
    return ov




def get_bboxes_from_lines(lines, img_size, gt_type):
    bboxes = []
    thumb_img_wid = img_size[0]
    thumb_img_hei = img_size[1]
    for line in lines:
        if line == '':
            break
        scale = 1.0
        vals = line.strip().split('\t')
        orgl_img_wid = float(vals[5])
        scale = thumb_img_wid / orgl_img_wid
        wid = float(vals[9])
        hei = float(vals[10])
        
        if gt_type == 0:
            x1 = float(vals[7])
            y1 = float(vals[8])
        if gt_type == 1:
            xc = float(vals[7]) #center x
            yc = float(vals[8]) #center y
            x1 = xc - wid / 2 
            y1 = yc - hei / 2
        x2 = x1 + wid - 1
        y2 = y1 + hei - 1
        x1 = int(np.clip(x1 * scale, 0.0, thumb_img_wid - 1))
        x2 = int(np.clip(x2 * scale, 0.0, thumb_img_wid - 1))
        y1 = int(np.clip(y1 * scale, 0.0, thumb_img_hei - 1))
        y2 = int(np.clip(y2 * scale, 0.0, thumb_img_hei - 1))
        bboxes.append([x1, y1, x2-x1+1, y2-y1+1, int(vals[12])])
    return bboxes



def check_bootstrapping_gt():
    #btsp_gt_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged_btsp1"
    btsp_gt_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged_btsp2"
    #rst_dir = "./work_data/faster-rcnn/btsp_vis/btsp1"
    rst_dir = "./work_data/faster-rcnn/btsp_vis/btsp2"
    cate2cls_file = "./data/HouzzDataCollection/HouzzData1/OrigBoxList/cat_table_purged.txt"
    img_dir = "./data/HouzzDataCollection/HouzzData1/Images"
    btsp_sig = "target_space_id"
    gt_type = 1 #xc, yc represent the center

    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir)

    file_list = glob.glob( os.path.join(btsp_gt_dir, "*.txt") )
    overlap_thresh = 0.5
    ct = 0

    cate2clsname = dict()
    with open(cate2cls_file, 'r') as fr:
        for line in fr:
            vals = line.strip().split('\t')
            cate2clsname[ int(vals[1]) ] = vals[2]

    for file_name in file_list:
        ct += 1
        print ct, '. ', file_name
        with open(file_name, 'r') as fr:
            content = fr.read()
        if btsp_sig not in content:
            #no bootstrapping bbox, skip
            continue
        basename = os.path.basename(file_name)
        img_name = os.path.join(img_dir, basename[:-4] + ".jpg")
        img_size = PIL.Image.open(img_name).size
        lines = content.split('\n')
        orig_lines = filter( (lambda x: btsp_sig not in x), lines)
        btsp_lines = filter( (lambda x: btsp_sig in x), lines)
        #bbox structure: [x_upleft, y_upleft, wid, hei, class]
        #print "lines"
        #embed()
        orig_bboxes = get_bboxes_from_lines(orig_lines, img_size, gt_type)
        btsp_bboxes = get_bboxes_from_lines(btsp_lines, img_size, gt_type)

        #visualize orig_bboxes in blue
        img = cv2.imread(img_name)[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12)) 
        ax.imshow(img, aspect='equal')
        for orig_bbox in orig_bboxes:
            ax.add_patch(
                plt.Rectangle( (orig_bbox[0], orig_bbox[1]), orig_bbox[2], orig_bbox[3], fill=False,
                                edgecolor='blue', linewidth=3.5) )
            ax.text( orig_bbox[0], orig_bbox[1] - 2,
                    '{:s}'.format(cate2clsname[orig_bbox[4]]),
                    bbox=dict(facecolor='black', alpha=0.5),
                    fontsize=14, color='white')

        for btsp_bbox in btsp_bboxes:
            bbox_color = 'green'
            for orig_bbox in orig_bboxes:
                ov = bboxes_overlap(btsp_bbox, orig_bbox)
                if ov >= overlap_thresh:
                    bbox_color = 'red'
                    break
            ax.add_patch(
                plt.Rectangle( (btsp_bbox[0], btsp_bbox[1]), btsp_bbox[2], btsp_bbox[3], fill=False,
                                edgecolor=bbox_color, linewidth=3.5) )
            ax.text( btsp_bbox[0], btsp_bbox[1] - 2,
                    'ov {:.2f}, {:s}'.format(ov, cate2clsname[btsp_bbox[4]]),
                    bbox=dict(facecolor='black', alpha=0.5),
                    fontsize=14, color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join( rst_dir, basename[:-4]+'.jpg'))
        plt.close()
        print "Figure drawn"




def move_file_by_time():
    #src_pattern = "./work_data/faster-rcnn/VGG16/dets_end2end_n0011_i98000_houzzdata1_val/t0.2/*.txt"
    src_pattern = "./work_data/faster-rcnn/VGG16/dets_end2end_n0011_i98000_houzzdata1_val/t0.2/*.jpg"
    dst_path = "./work_data/faster-rcnn/VGG16/dets_end2end_n0011_i98000_houzzdata1_test/t0.2"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    img_list = glob.glob(src_pattern)
    #filter by date
    #target_date = "Jan 12"
    #target_date = "Jan 27"
    target_date = "Feb 10"
    img_list = [ f for f in img_list if target_date in time.ctime(os.path.getctime(f)) ]
    ct = 0
    for item in img_list:
        ct += 1
        shutil.move(item, dst_path)
        print ct, ". ", item
    


def remove_squares_from_annotations():
    #src_list_file = "./data/HouzzDataCollection/HouzzData1/ImageLists/val.txt"
    #dst_list_file = "./data/HouzzDataCollection/HouzzData1/ImageLists/val_nosquare.txt"
    #src_list_file = "./data/HouzzDataCollection/HouzzData1/ImageLists/test.txt"
    #dst_list_file = "./data/HouzzDataCollection/HouzzData1/ImageLists/test_nosquare.txt"
    src_list_file = "./data/HouzzDataCollection/HouzzData1/ImageLists/train34000+6000+9000.txt"
    dst_list_file = "./data/HouzzDataCollection/HouzzData1/ImageLists/train34000+6000+9000_nosquare.txt"
    src_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged"
    dst_dir = "./data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged_nosquare"
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with open(src_list_file, 'r') as fr0:
        img_list = fr0.read().strip().split('\n')

    ct = 0
    boxes_tot = 0
    boxes_removed_tot = 0
    files_tot = len(img_list)
    files_removed_tot = 0
    list_names = []


    for space_id in img_list:
        ct += 1
        txt_name = space_id + '.txt' 
        src_txt = os.path.join(src_dir, txt_name)
        dst_txt = os.path.join(dst_dir, txt_name)
        print ct, ". ", src_txt
        boxes_num, boxes_removed, is_file_written = remove_squares_in_file(src_txt, dst_txt)
        boxes_tot += boxes_num
        boxes_removed_tot += boxes_removed
        if not is_file_written:
            files_removed_tot += 1
            continue
        list_names.append(space_id)

    print "Total boxes: ", boxes_tot
    print "Total boxes removed: ", boxes_removed_tot
    print "Total files: ", files_tot
    print "Total files removed: ", files_removed_tot
    with open(dst_list_file, 'w' ) as fw:
        fw.write('\n'.join(list_names))




def remove_squares_in_file(src_txt, dst_txt):
    content = []
    box_ct = 0
    removed_ct = 0
    is_file_written = True
    with open(src_txt, 'r') as fr:
        for line in fr:
            box_ct += 1
            vals = line.strip().split('\t')
            if vals[9] == vals[10]:
                removed_ct += 1
                continue
            else:
                content.append(line)

    if len(content) == 0:
        is_file_written = False
    else: 
        with open(dst_txt, 'w') as fw:
            fw.write(''.join(content))
    return (box_ct, removed_ct, is_file_written)




def map_class_label():
    src_dir = "/home/ubuntu/work/data/HouzzDataCollection/HouzzData1/annotationsAll/cate_purged_nosquare"
    dst_dir = "/home/ubuntu/work/data/HouzzDataCollection/HouzzData1/annotationsAll/cate_binary_nosquare"

    filelist = glob.glob( os.path.join(src_dir, '*.txt') )
    for src_pathname in filelist:
        basename = os.path.basename(src_pathname)
        dst_pathname = os.path.join(dst_dir, basename)
        merge(None, src_pathname, dst_pathname, fixed_target_value='2')



def merge_cate():
    ref_train_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039.txt"
    ref_val_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_holdout2000.txt"
    ref_test_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_testset5000.txt"
    ref_all_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015.txt"
    dst_train_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_trainset94039_cate_merged.txt"
    dst_val_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_holdout2000_cate_merged.txt"
    dst_test_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_testset5000_cate_merged.txt"
    dst_all_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/tag_boxes_11192015_cate_merged.txt"

    cate_merge_file = "./data/HouzzDataCollection/HouzzData0/OrigBoxList/cat_merge_map.txt"
    
    cate_merge_map = load_cate_merge_map(cate_merge_file)

    merge(cate_merge_map, ref_train_file, dst_train_file)
    merge(cate_merge_map, ref_val_file, dst_val_file)
    merge(cate_merge_map, ref_test_file, dst_test_file)
    merge(cate_merge_map, ref_all_file, dst_all_file)






def ensemble_results1():
    nms_thresh_all = 0.7
    src1_dir = "./work_data/faster-rcnn/VGG16/dets_end2end_n0016_train_nosquare_i90200_houzzdata1_val_nosquare/t0.2"
    #src2_dir = "./work_data/faster-rcnn/VGG16/dets_end2end_n0016_train_nosquare_i80000_houzzdata1_val_nosquare/t0.2"
    src2_dir = "./work_data/faster-rcnn/VGG16/dets_end2end_n0011_i98000_houzzdata1_val/t0.2"
    rst_dir = "./work_data/faster-rcnn/VGG16/dets_end2end_n0016_i90200-n0011_i80000_train_nosquare_houzzdata1_val_nosquare_nms{:.1f}/t0.2".format(nms_thresh_all)
    cfg.GPU_ID = 1
    print "Using GPU {}".format(cfg.GPU_ID)

    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir)
    
    file_list = glob.glob(os.path.join(src1_dir, "*.txt")) 
    ct = 0
    for filename1 in file_list:
        basename = os.path.basename(filename1)
        if basename == 'detection_record.txt':
            continue
        ct += 1
        print "{}.  {}".format(ct, basename)
        filename2 = os.path.join(src2_dir, basename)
        dets1 = read_dets(filename1)
        dets2 = read_dets(filename2)
        all_dets = np.vstack( (dets1, dets2) ).astype(np.float32) #float32 required by nms
        all_keep = nms(all_dets[:, [0,1,2,3,5]], nms_thresh_all)
        all_dets = all_dets[all_keep, :]
        write_dets(os.path.join(rst_dir, basename), all_dets)



def read_dets(filename):
    content_list = []
    with open(filename, 'r') as fr:
        for line in fr:
            vals = line.strip().split()
            x1 = float(vals[0])
            y1 = float(vals[1])
            wid = float(vals[2])
            hei = float(vals[3])
            cls = float(vals[4])
            score = float(vals[5])
            x2 = x1 + wid - 1
            y2 = y1 + hei - 1
            det = [x1, y1, x2, y2, cls, score]
            content_list.append(det)
    return np.array(content_list)




def write_dets(filename, dets):
    content_list = []
    for det in dets:
        content_list.append( "{:d} {:d} {:d} {:d} {:d} {:.4f}".format(\
                int(det[0]), int(det[1]), int(det[2]-det[0]+1), int(det[3]-det[1]+1), int(det[4]), det[5]))
    content = '\n'.join(content_list)
    with open(filename, 'w') as fw:
        fw.write(content)
    return 
            






if __name__ == "__main__":
#Analysis before partitioning sets
    ##find_space_overlap()

#Necessary steps to prepare data
    #merge_cate()
    #partition_sets()
    #split_box_list_by_space()
    #download_images()

#Analysis
    ##find_space_overlap()
    ##cate_dstb_for_sets()
    ##get_sample_training_images()
    ##show_classes()

#Analysis for purge
    #show_merge_names_and_counts()

#Generate purged data
    ##Copy cat_merge_map_show_names.txt to cat_purge_map_draft, and manually make changes 
    #check_purge_map_draft()
    #gen_purge_map()
    #purge_cates()
    #partition_purged_sets()
    #split_box_list_of_purged_cates()

#Process new data: image_tag_bounds
    #purge_cates_for_new_data()
    #split_box_list_of_purged_cates_for_new_data()
    #gen_image_list()
    #download_images_for_new_data()
    #check_oldnew_overlap_and_get_new_train_list()
    #cate_dstb_for_sets()

#got another batch of image_tag_bounds data
    ##find_space_overlap_houzzdata1_different_dates()
    #purge_cates_for_new_data()
    #split_box_list_of_purged_cates_for_new_data()
    #gen_image_list()
    #download_images_for_new_data()
    #check_oldnew_overlap_and_get_new_train_val_test_list()

#remove squares boxes
    #remove_squares_from_annotations()

#create binary class data consisting of just foreground and background class
    map_class_label()



#temp use
    #create_small_set()
    #convert_gt_to_det_format()
    #plot_training_error_curve()    
    #plot_training_error_curve2()    
    #draw_training_gt_boxes()
    #check_image_lists_overlap()
    #get_bootstrapping_gt_from_final_prediction()
    #get_bootstrapping_gt_from_detection()
    #check_bootstrapping_gt()
    #move_file_by_time()
    #ensemble_results1()
    download_temp_images()

    

