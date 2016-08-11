# --------------------------------------------------------
# lekang@houzz.com
# --------------------------------------------------------

#import datasets.pascal_voc
#import datasets.houzz_data
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
#import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
import PIL
from IPython import embed
from fast_rcnn.config import cfg


class houzz_data(imdb):
    #def __init__(self, image_set, year, devkit_path=None):
    def __init__(self, image_set, set_version, devkit_path=None):
        print "Initializing Houzz data"
        #imdb.__init__(self, 'voc_' + year + '_' + image_set)
        imdb.__init__(self, "houzzdata" + set_version + "_" + image_set)
        self._set_version = set_version
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'HouzzData' + self._set_version)
        #self._cate_merge_map_file = os.path.join(self._data_path, "OrigBoxList", "cat_merge_map.txt")
        self._cate_merge_map_file = os.path.join(self._data_path, "OrigBoxList", "cat_purge_map.txt")
        #self._cate_table_file = os.path.join(self._data_path, "OrigBoxList", "cat_table_merged.txt")
        self._cate_table_file = os.path.join(self._data_path, "OrigBoxList", "cat_table_purged.txt")
        #self._classes = ('__background__', # always index 0
                         #'aeroplane', 'bicycle', 'bird', 'boat',
                         #'bottle', 'bus', 'car', 'cat', 'chair',
                         #'cow', 'diningtable', 'dog', 'horse',
                         #'motorbike', 'person', 'pottedplant',
                         #'sheep', 'sofa', 'train', 'tvmonitor')
        #self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        # houzzdata specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'top_k'       : 2000,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None}

        #assert os.path.exists(self._devkit_path), \
                #'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._devkit_path), \
                'Houzz data collection path does not exist: {}'.format(self._devkit_path)
        #assert os.path.exists(self._data_path), \
                #'Path does not exist: {}'.format(self._data_path)
        assert os.path.exists(self._data_path), \
                'Dataset path does not exist: {}'.format(self._data_path)
        assert os.path.exists(self._cate_merge_map_file)
        self.cate_merge_map = self._load_cate_merge_map()
        assert os.path.exists(self._cate_table_file)
        self._classes, self._class_to_ind, self._categoryID_to_ind = self._load_classes()
        print "Finished initializing houzz data"
        #embed()


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])


    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        #image_path = os.path.join(self._data_path, 'JPEGImages',
                                  #index + self._image_ext)
        image_path = os.path.join(self._data_path, 'Images', 
                                    index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    

    def _load_classes(self):
        """
        Load classes, class to index map, and category id to index map
        """
        class_to_ind = {}
        categoryID_to_ind = {}
        ct = 1
        with open(self._cate_table_file, 'r') as fh:
            for line in fh:
                vals = line.strip().split('\t')
                categoryID_to_ind[int(vals[1])] = ct                
                class_to_ind[vals[2]] = ct
                ct += 1
        class_to_ind['__background__'] = 0
        #categoryID_to_ind[0] = 0
        classes = tuple(class_to_ind.keys())
        return classes, class_to_ind, categoryID_to_ind




    def _load_cate_merge_map(self):
        """
        Load category merge map
        """
        cate_merge_map = {}
        with open(self._cate_merge_map_file, 'r') as fh:
            for line in fh:
                vals = line.strip().split()
                cate_merge_map[int(vals[0])] = int(vals[1])                
        return cate_merge_map
                


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        #image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      #self._image_set + '.txt')
        image_set_file = os.path.join(self._data_path, 'ImageLists', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Image list file does not exist: {}'.format(image_set_file)
        ct = 0
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
            ct += 1
        print "{} lines read from {}".format(ct, image_set_file)
        return image_index

    def _get_default_path(self):
        """
        return the default path where pascal voc is expected to be installed.
        """
        #return os.path.join(datasets.ROOT_DIR, 'data', 'vocdevkit' + self._year)
        #temp_ROOT_DIR = "/home/ubuntu/py-faster-rcnn"
        #return os.path.join(datasets.ROOT_DIR, 'data', 'HouzzDataCollection') #not sure
        return os.path.join(cfg.DATA_DIR, 'HouzzDataCollection') #not sure

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        #print "Not using cache"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_houzz_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        raise Exception("selective_search_roidb: to be implemented")

        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        #raise Exception("rpn_roidb: To be implemented")

        #if int(self._year) == 2007 or self._image_set != 'test':
        if int(self._set_version) == 0 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        raise Exception("_load_selective_search_roidb: to be implemented")

        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_houzz_annotation(self, index):
        """
        #Load image and bounding boxes info from XML file in the PASCAL VOC
        #format.
        Load image and bounding boxes info from txt files.
        """
        #filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        #filename = os.path.join(self._data_path, 'Annotations/cate_merged', index + '.txt')
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')

        # print 'Loading: {}'.format(filename)

        #def get_data_from_tag(node, tag):
            #return node.getElementsByTagName(tag)[0].childNodes[0].data

        #with open(filename) as f:
            #data = minidom.parseString(f.read())

        #objs = data.getElementsByTagName('object')
        #if not self.config['use_diff']:
            ## Exclude the samples labeled as difficult
            #non_diff_objs = [obj for obj in objs
                             #if int(get_data_from_tag(obj, 'difficult')) == 0]
            #if len(non_diff_objs) != len(objs):
                #print 'Removed {} difficult objects' \
                    #.format(len(objs) - len(non_diff_objs))
            #objs = non_diff_objs
        #num_objs = len(objs)
        num_lines = sum(1 for line in open(filename))        
        boxes = np.zeros((num_lines, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_lines), dtype=np.int32)
        overlaps = np.zeros((num_lines, self.num_classes), dtype=np.float32)

        ## Load object bounding boxes into a data frame.
        #for ix, obj in enumerate(objs):
            ## Make pixel indexes 0-based
            #x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            #y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            #x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            #y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            #cls = self._class_to_ind[
                    #str(get_data_from_tag(obj, "name")).lower().strip()]
            #boxes[ix, :] = [x1, y1, x2, y2]
            #gt_classes[ix] = cls
            #overlaps[ix, cls] = 1.0
        ct = 0
        thumb_img_size = PIL.Image.open(self.image_path_from_index(index)).size
        thumb_img_wid = thumb_img_size[0]
        thumb_img_hei = thumb_img_size[1] 
        scale = 1.0
        if self._set_version == '0':
            with open(filename, 'r') as fh:
                for line in fh:
                    vals = line.strip().split('\t')
                    orgl_img_wid = float(vals[5])
                    scale = thumb_img_wid / orgl_img_wid
                    x1 = float(vals[7])
                    y1 = float(vals[8])
                    x2 = float(vals[9]) + x1 - 1
                    y2 = float(vals[10]) + y1 - 1
                    x1 = np.clip(x1 * scale, 0.0, thumb_img_wid - 1)
                    x2 = np.clip(x2 * scale, 0.0, thumb_img_wid - 1)
                    y1 = np.clip(y1 * scale, 0.0, thumb_img_hei - 1)
                    y2 = np.clip(y2 * scale, 0.0, thumb_img_hei - 1)
                    categoryID = int(vals[-3])
                    if categoryID not in self._categoryID_to_ind:
                        print "Likely keyvalue error."
                        print "vals: ", vals
                        categoryID = 2002 # wrong keys mapped to "tabel top"
                    cls = self._categoryID_to_ind[categoryID]
                    boxes[ct, :] = [x1, y1, x2, y2]
                    gt_classes[ct] = cls
                    overlaps[ct, cls] = 1.0
                    ct += 1
        elif self._set_version == '1':
            with open(filename, 'r') as fh:
                for line in fh:
                    vals = line.strip().split('\t')
                    orgl_img_wid = float(vals[5])
                    scale = thumb_img_wid / orgl_img_wid
                    xc = float(vals[7]) #center x
                    yc = float(vals[8]) #center y
                    wid = float(vals[9])
                    hei = float(vals[10])
                    x1 = xc - wid / 2
                    y1 = yc - hei / 2
                    x2 = x1 + wid - 1
                    y2 = y1 + hei - 1
                    x1 = np.clip(x1 * scale, 0.0, thumb_img_wid - 1)
                    x2 = np.clip(x2 * scale, 0.0, thumb_img_wid - 1)
                    y1 = np.clip(y1 * scale, 0.0, thumb_img_hei - 1)
                    y2 = np.clip(y2 * scale, 0.0, thumb_img_hei - 1)
                    categoryID = int(vals[-3])
                    if categoryID not in self._categoryID_to_ind:
                        print "Likely keyvalue error."
                        print "vals: ", vals
                        categoryID = 2002 # wrong keys mapped to "tabel top"
                    cls = self._categoryID_to_ind[categoryID]
                    boxes[ct, :] = [x1, y1, x2, y2]
                    gt_classes[ct] = cls
                    overlaps[ct, cls] = 1.0
                    ct += 1
        else:
            raise ValueError("Incorrect set version for parsing annotation!")

        boxes = boxes[:ct]
        gt_classes = gt_classes[:ct]
        overlaps = overlaps[:ct]

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _get_comp_id(self):
        raise Exception("To be implemented")
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id


    def _write_houzzdata_results_file(self, all_boxes):
        raise Exception("To be implemented")

        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        raise Exception("To be implemented")

        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        raise Exception("To be implemented")
        comp_id = self._write_houzzdata_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        raise Exception("To be implemented")
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.houzz_data import houzz_data
    d = houzz_data('train', '1')
    res = d.roidb
    from IPython import embed; embed()
