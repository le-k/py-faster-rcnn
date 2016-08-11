#!/usr/bin/env python
import _init_paths
import os
import numpy as np
import cPickle as pkl
from IPython import embed
import itertools
from PIL import Image, ImageDraw
from utils.timer import Timer
import argparse



def get_map_among_cate_cls_name(cate2cls_file):
        cate2cls = {}
        cls2cate = {}
        cls2name = {}

        with open(cate2cls_file, 'r') as fh:
            for line in fh:
                vals = line.strip().split('\t')
                cls_ind = int(vals[0])
                cate_id = vals[1]
                name = vals[2]
                cate2cls[cate_id] = cls_ind
                cls2cate[cls_ind] = cate_id
                cls2name[cls_ind] = name
        return cate2cls, cls2cate, cls2name



def create_sbs_img(img_basename, gt_bboxes):
    #create a side-by-side image: left half for detected boxes, right half for gt boxes
    #img = Image.open(img_basename)
    #gt_img = img.copy()
    #draw = ImageDraw.Draw(gt_img)
    #for i in xrange(len(gt_bboxes)):
        #draw.rectangle(list(gt_bboxes[i,:]), outline = 'green')
    #del draw
    #wid, hei = img.size
    #sbs_img = Image.new('RGB', (wid * 2,  hei))
    #sbs_img.paste(img, (0,0))
    #sbs_img.paste(gt_img, (wid, 0))

    img = Image.open(img_basename)
    wid, hei = img.size
    gt_img = img.copy()
    sbs_img = Image.new('RGB', (wid * 2,  hei))
    sbs_img.paste(img, (0,0))
    sbs_img.paste(gt_img, (wid, 0))
    draw = ImageDraw.Draw(sbs_img)
    color = 'green' 
    for i in xrange(len(gt_bboxes)):
        draw.rectangle(list(gt_bboxes[i,:] + np.array([wid, 0, wid, 0])), outline = color)
    del draw

    ###
    #print "create_sbs_img"
    #embed()
    ###
    return sbs_img


    
def draw_box_on_img( sbs_img, box, is_hit, is_det):
    draw = ImageDraw.Draw(sbs_img)
    sbs_wid, sbs_hei = sbs_img.size
    color = 'green' 
    shift = 0
    if is_det: #detected boxes 
        color = 'red'
    else: #gt boxes
        color = 'green'
        shift = sbs_wid/2
    if is_hit: #correct box, either detected or gt
        color= 'blue'
    draw.rectangle(list(box + np.array([shift, 0, shift, 0])), outline = color)
    del draw
    ###
    #print "draw_box_on_img"
    #embed()
    return 


def eval_detection( predict_dir, gtruth_dir, test_img_list, cate2cls_file, report_file, gt_type, ignore_cate, conf_thresh = 0.2, img_dir = None,
        sbs_img_dir = None, blacklist_file = None, optional_cache_file = None):
    
    defaultIOUthr = 0.5
    pixelTolerance = 10
    img_ext = ".jpg"
    bbox_ext = ".txt"
    timer0 = Timer()
    load_gt_cache = False
    class_total = 137 #excluding background

    cate2cls, cls2cate, cls2name = get_map_among_cate_cls_name(cate2cls_file)    
    all_cls = cls2cate.keys()
    gt_stats = dict( zip(all_cls, [0] * len(all_cls)) ) #box counts of gt in each class
    gt_hit = gt_stats.copy() #detected gt boxes for each class
    gt_hit_rate = gt_stats.copy() #percentage

    screen_squares = True
    square_removed = 0
    square_remained = 0

    if load_gt_cache and (optional_cache_file is not None) and (os.path.exists(optional_cache_file)):
        print 'eval_detection :: loading cached ground truth'
        timer0.tic()
        gt_img_ids,gt_obj_labels,gt_obj_bboxes,gt_obj_thr,box_num_per_class =  pkl.load(
                optional_cache_file);
        timer0.toc()
        print 'eval_detection :: loading cached ground truth took {:.1f} seconds'.format(
                timer0.total_time)
        if 'gt_obj_img_ids' in locals():
            print('eval_detection :: loaded cache' )
        else:
            load_gt_cache = False

    if not load_gt_cache:
        print 'eval_detection :: loading ground truth'
        timer1 = Timer()
        timer1.tic()
    
        #[img_basenames gt_img_ids] = textread(eval_file,'%s %d');
        with open(test_img_list, 'r') as fh0:
            img_basenames = [line.strip() + img_ext for line in fh0.readlines()]
        num_imgs = len(img_basenames);
        gt_img_ids = range(num_imgs)
        #gt_obj_labels = [None] * num_imgs #cell(1,num_imgs);
        #gt_obj_bboxes = [None] * num_imgs #cell(1,num_imgs);
        #gt_obj_thr = [None] * num_imgs #cell(1,num_imgs);
        gt_obj_labels = {}
        gt_obj_bboxes = {}
        gt_obj_thr = {}
        box_num_per_class = {}
        timer2 = Timer()
        timer2.tic()
        last_total_time = timer2.total_time
        for i in xrange(num_imgs):
            timediff = timer2.toc(False)
            if timer2.total_time > 10: 
                print '              :: on {:d} of {:d}'.format(i,num_imgs)
            timer2.tic();
            bb_list = []
            label_list = []
            with open( os.path.join( gtruth_dir, img_basenames[i][:-4] + bbox_ext), 'r' ) as fh1:
                img_name = os.path.join(img_dir, img_basenames[i])
                thumb_img_wid, thumb_img_hei = Image.open(img_name).size
                box_ct = 0
                for line in fh1:
                    if gt_type == 0:        
                        vals = line.strip().split('\t')
                        cate_id = vals[-3]
                        cls_ind = cate2cls[cate_id]
                        orgl_img_wid = float(vals[5])
                        scale = thumb_img_wid / orgl_img_wid
                        x1_orig = float(vals[7])  
                        y1_orig = float(vals[8])
                        x2_orig = float(vals[9]) + x1_orig - 1
                        y2_orig = float(vals[10]) + y1_orig - 1
                        gt_w = float(vals[9]) * scale
                        gt_h = float(vals[10]) * scale
                        x1 = np.clip(x1_orig * scale, 0.0, thumb_img_wid - 1)
                        x2 = np.clip(x2_orig * scale, 0.0, thumb_img_wid - 1)
                        y1 = np.clip(y1_orig * scale, 0.0, thumb_img_hei - 1)
                        y2 = np.clip(y2_orig * scale, 0.0, thumb_img_hei - 1)
                    elif gt_type == 1:
                        vals = line.strip().split('\t')
                        cate_id = vals[-3]
                        cls_ind = cate2cls[cate_id]
                        orgl_img_wid = float(vals[5])
                        scale = thumb_img_wid / orgl_img_wid
                        xc = float(vals[7]) #center x
                        yc = float(vals[8]) #center y
                        wid = float(vals[9])
                        hei = float(vals[10])
                        if wid == hei and screen_squares:
                            if box_ct > 0:
                                square_removed += 1
                                continue
                            else:
                                square_remained += 1

                        x1 = xc - wid / 2
                        y1 = yc - hei / 2
                        x2 = x1 + wid - 1
                        y2 = y1 + hei - 1
                        x1 = np.clip(x1 * scale, 0.0, thumb_img_wid - 1)
                        x2 = np.clip(x2 * scale, 0.0, thumb_img_wid - 1)
                        y1 = np.clip(y1 * scale, 0.0, thumb_img_hei - 1)
                        y2 = np.clip(y2 * scale, 0.0, thumb_img_hei - 1)
                    else:
                        print "Wrong ground truth type. "
                        exit(1)
                    label_list.append(cls_ind)
                    bb_list.append([x1, y1, x2, y2])
                    if cls_ind in box_num_per_class:
                        box_num_per_class[cls_ind] += 1
                    else:
                        box_num_per_class[cls_ind] = 1
                    box_ct += 1
            bbs = np.array(bb_list)
            gt_w = bbs[:,2] - bbs[:,0] + 1 #column vector
            gt_h = bbs[:,3] - bbs[:,1] + 1 #column vector
            thr = (gt_w * gt_h) / ((gt_w+pixelTolerance)*(gt_h+pixelTolerance)) #element wise
            gt_obj_thr[i] = np.fmin(defaultIOUthr,thr) #element wise
            gt_obj_bboxes[i] = bbs
            gt_obj_labels[i] = label_list 
            for cls_ind in label_list:
                gt_stats[cls_ind] += 1

        timer2.toc()
        print 'eval_detection :: loading ground truth took {:.1f} seconds'.format(timer2.total_time)
        if optional_cache_file is not None:
            print "eval_detection :: saving cache in {}".format(optional_cache_file)
            with open(optional_cache_file, 'w') as fh3:
                pkl.dump((gt_img_ids,gt_obj_labels,gt_obj_bboxes,gt_obj_thr,box_num_per_class), fh3,
                        protocol = pkl.HIGHEST_PROTOCOL)

    blacklist_img_id = None
    blacklist_label = None
    if (blacklist_file is not None) and os.path.exist(blacklist_file):
        blacklist_img_id = []
        blacklist_label = []
        with open(blacklist_file, 'r') as fh4:
            for line in fh4:
                vals = line.strip().split()
                blacklist_img_id.append(vals[0])
                blacklist_label.append(cate2cls[vals[1]])
        blacklist_img_id = np.array(blacklist_img_id) 
        blacklist_label = np.array(blacklist_label)
        print 'eval_detection :: blacklisted {} image/object pairs'.format(len(blacklist_label))
    else:
        print 'eval_detection :: no blacklist'

    #load detection results
    pred_obj_labels = {}
    pred_obj_bboxes = {}
    pred_obj_scores = {}
    for i in xrange(len(img_basenames)):
        predict_file = os.path.join(predict_dir, img_basenames[i][:-4] + bbox_ext)
        if os.path.exists(predict_file):
            bb_list = []
            label_list = []
            score_list = []
            with open(predict_file, 'r') as fh5:
                ct_line = 0
                for line in fh5:
                    ct_line += 1
                    vals = line.strip().split()
                    x0 = float(vals[0])
                    y0 = float(vals[1])
                    wid = float(vals[2])
                    hei = float(vals[3])
                    cls_ind = int(vals[4])
                    score = float(vals[5])
                    if score < conf_thresh:#filter out low score objects
                        continue
                    bb_list.append([x0, y0, x0 + wid - 1, y0 + hei - 1])
                    label_list.append(cls_ind)
                    score_list.append(score)

            bb_array = np.array(bb_list)
            label_array = np.array(label_list)
            score_array = np.array(score_list)
            pred_obj_labels[i] = np.array(label_array)
            pred_obj_bboxes[i] = np.array(bb_array)
            pred_obj_scores[i] = np.array(score_array)
        else:
            pass

    tp_dict = {}
    fp_dict = {}
    timer3 = Timer()
   
    #compute overlap and decide true/false positive 
    for i in xrange(len(img_basenames)):
        timediff = timer3.toc(False)
        if timediff > 60:
            print '               :: on {} of {}'.format(i, len(gt_img_ids))
        timer3.tic()

        gt_labels = gt_obj_labels[i]
        gt_bboxes = gt_obj_bboxes[i]
        gt_thr = gt_obj_thr[i]
        num_gt_obj = len(gt_labels)
        gt_detected = np.array( [0.0] * num_gt_obj) 
        sbs_img_name = None
        if sbs_img_dir is not None:
            sbs_img_name = os.path.join(sbs_img_dir, img_basenames[i])
            sbs_img =  create_sbs_img(os.path.join(img_dir, img_basenames[i]),\
                    gt_bboxes)

        #if not (i in gt_obj_labels and i in pred_obj_labels):
        if not(i in pred_obj_labels):
            if sbs_img_dir is not None:
                sbs_img.save(sbs_img_name) 
            continue

        blacklisted_obj = None
        if blacklist_img_id is not None:
            bSameImg = blacklist_img_id == img_basenames[i]
            blacklisted_obj = blacklist_label[bSameImg]

        pred_labels = pred_obj_labels[i]
        pred_bboxes = pred_obj_bboxes[i]

        num_obj = len(pred_labels)
        tp = [0.0] * num_obj
        fp = [0.0] * num_obj
        for j in xrange(num_obj):
            if (blacklisted_obj is not None) and any( pred_labels[j] == blacklisted_obj):
                continue #just ignore this detection
            bb = pred_bboxes[j, :]
            ovmax = float('-inf')
            kmax = -1
            for k in xrange(num_gt_obj):
                if (not ignore_cate) and (pred_labels[j] != gt_labels[k]): #same label
                    continue
                if gt_detected[k] > 0:
                    continue
                bbgt = gt_bboxes[k, :]
                bi = np.array( [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),\
                        min(bb[2], bbgt[2]), min(bb[3], bbgt[3])] )
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap as area of intersection over area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) +\
                        (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) -\
                        iw * ih
                    ov = iw * ih / ua
                    # makes sure that this object is detected according
                    # to its individual threshold
                    if (ov >= gt_thr[k]) and (ov > ovmax):
                        ovmax = ov
                        kmax = k
            if kmax >= 0:
                tp[j] = 1
                gt_detected[kmax] = 1
                gt_hit[gt_labels[kmax]] += 1
                if sbs_img_dir is not None:
                    draw_box_on_img(sbs_img, bb, is_hit = True, \
                        is_det = True) #highlight detected box
                    draw_box_on_img(sbs_img, gt_bboxes[kmax,:], is_hit = True, \
                        is_det = False) #highlight gt box
            else:
                fp[j] = 1
                if sbs_img_dir is not None:
                    draw_box_on_img(sbs_img, bb, is_hit = False, \
                        is_det = True) #highlight detected box

        if sbs_img_dir is not None:
            sbs_img.save(sbs_img_name)
        tp_dict[i] = tp
        fp_dict[i] = fp

        for k in xrange(num_gt_obj):
            label = gt_labels[k]
            #remove blacklisted objects from consideration as positive examples (ground truth)
            if (blacklisted_obj is not None) and any(label == blacklisted_obj):
                box_num_per_class[label] = box_num_per_class[label] - 1

    timer3.toc()
    print 'eval_detection :: accumulating took [:0.1f] seconds'.format(timer3.total_time)
    print 'eval_detection :: computing ap'
    
    #compute precision, recall, and ap
    timer4 = Timer()
    timer4.tic()
    #concatente
    tp_all = np.array( list( itertools.chain.from_iterable(tp_dict.values()) ) ).astype('float')
    fp_all = np.array( list( itertools.chain.from_iterable(fp_dict.values()) ) ).astype('float')
    pred_obj_labels_all = np.array( list( itertools.chain.from_iterable(pred_obj_labels.values()) ) )
    scores_all = np.array( list( itertools.chain.from_iterable(pred_obj_scores.values()) ) )
    
    (ap_all, recall_all, precision_all), (ap_cls, recall_cls, precision_cls) = \
            compute_ap_recall_precision(tp_all, fp_all, pred_obj_labels_all, scores_all, \
                    box_num_per_class, class_total, ignore_cate)

    with open(report_file, 'a') as fw:
        print "Total gt boxes: {}\n".format(sum(box_num_per_class.values())) 
        print "Removed squares tot: {}\n".format(square_removed)
        print "Remaining squares: {}\n".format(square_remained)
        fw.write("Total gt boxes: {}\n".format(sum(box_num_per_class.values())) )
        fw.write( "Removed squares: {}\n".format(square_removed))
        fw.write( "Remaining squares: {}\n".format(square_remained))
        title = "Ignore categories:" if ignore_cate else \
                "Overall"
        fw.write( "{}:\n".format(title) + \
                ("\tap:\t{:.3f}\n" + \
                "\trecall:\t{:.3f}\n" + \
                "\tprecision:\t{:.3f}\n").format(ap_all, recall_all, precision_all) )
        print "{}:\n".format(title) + \
                ("\tap:\t{:.3f}\n" + \
                "\trecall:\t{:.3f}\n" + \
                "\tprecision:\t{:.3f}\n").format(ap_all, recall_all, precision_all) 
        if ignore_cate:
            for key in gt_stats.keys():
                if gt_stats[key] > 0:
                    gt_hit_rate[key] = float(gt_hit[key]) / gt_stats[key]
                else:
                    gt_hit_rate[key] = -1.0
            #keys, hit_rates = zip(*gt_hit_rate.items())
            #keys = np.array(keys)
            #hit_rates = np.array(hit_rates)
            #inds = np.argsort(hit_rates)
            cls_inds, box_nums = zip(*box_num_per_class.items())
            cls_inds = np.array(cls_inds)
            box_nums = np.array(box_nums)
            inds = np.argsort(box_nums)[::-1]
            sorted_cls_inds = cls_inds[inds]

            fw.write( "For each class:\n")
            #for i in xrange(class_total):
                #c = i + 1
            ct = 0
            fw.write("Absence classes:\n")
            #for c in sorted_keys:
                #if gt_stats[c] > 0:
                    #continue
            for c in xrange(1, class_total + 1):
                if c in box_num_per_class:
                    continue
                ct += 1
                fw.write( "{}. class {}, category id {}, name [{}]:\n".format(ct, c, cls2cate[c], cls2name[c]))
            ct = 0
            fw.write("Present classess:\n")
            for c in sorted_cls_inds:
                #if gt_stats[c] == 0:
                if c not in box_num_per_class:
                    continue
                ct += 1
                #fw.write( "{}. class {}, category id {}, name [{}]:\n".format(ct, c, cls2cate[c], cls2name[c]) + \
                    #"\thit/recall:\t{:.3f} ({:d}/{:d})\n".format(gt_hit_rate[c], gt_hit[c], gt_stats[c]) )
                fw.write( "{}. class {}, category id {}, name [{}]:\n".format(ct, c, cls2cate[c], cls2name[c]) + \
                    "\thit/recall:\t{:.3f} ({:d}/{:d})\n".format(gt_hit_rate[c], gt_hit[c], box_num_per_class[c]) )
            fw.write( "\n\n" )
        else:
            inds = np.argsort(recall_cls)[::-1]
            cls_inds = np.arange(class_total + 1)[inds]
            recall_cls = recall_cls[inds]
            ap_cls = ap_cls[inds]
            precision_cls = precision_cls[inds]
            fw.write( "For each present class:\n")
            #for i in xrange(class_total):
                #c = i + 1
            ct = 0
            for c in cls_inds:
                if c == 0:
                    continue
                ct += 1
                fw.write( "{}. class {}, category id {}, name [{}]:\n".format(ct, c, cls2cate[c], cls2name[c]) + \
                        ("\tap:\t{:.3f}\n" + \
                        "\trecall:\t{:.3f}\n" + \
                        "\tprecision:\t{:.3f}\n").format(ap_cls[c], recall_cls[c], precision_cls[c]) )
            fw.write( "\n\n" )




def compute_ap_recall_precision(tp_all, fp_all, pred_obj_labels_all, scores_all, box_num_per_class, class_total, ignore_cate):
    ind = np.argsort(scores_all, )[::-1] #decending
    tp_all = tp_all[ind]
    fp_all = fp_all[ind]
    pred_obj_labels_all = pred_obj_labels_all[ind]

    tot_gt_obj = sum( box_num_per_class.values())
    tp_all_cumsum = np.cumsum(tp_all)
    fp_all_cumsum = np.cumsum(fp_all)
    recall_series_all = tp_all_cumsum / tot_gt_obj
    precision_series_all = tp_all_cumsum / (tp_all_cumsum + fp_all_cumsum)
    recall_all = recall_series_all[-1]
    precision_all = precision_series_all[-1]
    ap_all = VOCap(recall_series_all, precision_series_all)

    if ignore_cate:
        return (ap_all, recall_all, precision_all), (None, None, None)

    recall_series_cls = {}
    precision_series_cls = {}
    recall_cls = np.zeros(class_total + 1)
    precision_cls = np.zeros(class_total + 1)
    ap_cls = np.zeros(class_total + 1) #ap[i] for class i, ap[0] meaningless
    ap_cls[0] = -1.0
    for k in xrange(class_total):
        cls_ind = k + 1
        if cls_ind in box_num_per_class:
            inds = pred_obj_labels_all == cls_ind
            if any(inds):
                tp = np.cumsum(tp_all[inds])
                fp = np.cumsum(fp_all[inds])
                recall_series_cls[cls_ind] = tp / box_num_per_class[cls_ind] 
                precision_series_cls[cls_ind] = tp / (fp + tp)
                recall_cls[cls_ind] = recall_series_cls[cls_ind][-1]
                precision_cls[cls_ind] = precision_series_cls[cls_ind][-1]
                ap_cls[cls_ind] = VOCap(recall_series_cls[cls_ind], precision_series_cls[cls_ind])
            else:
                recall_series_cls[cls_ind] = [0.0]
                precision_series_cls[cls_ind] = [0.0]
                recall_cls[cls_ind] = 0.0
                precision_cls[cls_ind] = 0.0
                ap_cls[cls_ind] = 0.0
        else:
            recall_series_cls[cls_ind] = -1.0
            precision_series_cls[cls_ind] = -1.0
            recall_cls[cls_ind] = -1.0
            precision_cls[cls_ind] = -1.0
            ap_cls[cls_ind] = -1.0

    return (ap_all, recall_all, precision_all), (ap_cls, recall_cls, precision_cls)


               


def VOCap(recall, precision):
    #mrec=[0 ; rec ; 1];
    #mpre=[0 ; prec ; 0];
    #for i=numel(mpre)-1:-1:1
        #mpre(i)=max(mpre(i),mpre(i+1));
    #i=find(mrec(2:end)~=mrec(1:end-1))+1;
    #ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    mrcl = np.append( [0.0], np.append( recall, [1.0]) )
    mprc = np.append( [0.0], np.append( precision, [0.0]) )
    for i in reversed( xrange(len(mprc)- 1) ):
        mprc[i] = max(mprc[i], mprc[i+1])
    j = np.where(np.not_equal(mrcl[1:], mrcl[:-1]))[0] + 1
    ap = sum( (mrcl[j] - mrcl[j - 1]) * mprc[j] )
    
    return ap
    



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
    print "good here"
    args = parse_args()
    predict_dir = args.predict_dir
    conf_thresh = args.conf_thresh
    gt_type = args.gt_type
    gtruth_dir = args.gtruth_dir
    test_img_list = args.test_img_list
    img_dir = args.img_dir
    cate2cls_file = args.cate2cls_file
    report_file = args.report_file
    sbs = args.sbs

    ignore_cate = True
    sbs_img_dir = None
    if sbs :
        sbs_img_dir = predict_dir + "_sbs_imgs" if not ignore_cate else \
                    predict_dir + "_sbs_imgs_ignore_cate"

    if sbs_img_dir is not None and (not os.path.isdir(sbs_img_dir)):
        os.makedirs(sbs_img_dir)
    assert os.path.isdir(predict_dir), "predict_dir does not exist!"
    

    blacklist_file = None
    #optional_cache_file = "./data/HouzzDataCollection/HouzzData0/Annotations/test_annotation_cache.pkl"
    optional_cache_file = None
    #Write/append category-ignorant results. Note that it opens file by appending.
    #So do delete old files before generate new files
    eval_detection( predict_dir, gtruth_dir, test_img_list, cate2cls_file, report_file, gt_type, \
            ignore_cate, conf_thresh, img_dir, sbs_img_dir, blacklist_file, optional_cache_file)
    #append category-aware results to category-ignorant ones 
    ignore_cate = False
    sbs_img_dir = None
    eval_detection( predict_dir, gtruth_dir, test_img_list, cate2cls_file, report_file, gt_type, \
            ignore_cate, conf_thresh, img_dir, sbs_img_dir, blacklist_file, optional_cache_file)
