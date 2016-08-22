#!/usr/bin/env python
"""
Generate the residual learning network (resnet) + faster rcnn
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from IPython import embed

freeze_one_param = '''\
  param { lr_mult: 0 decay_mult: 0 }
'''


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--solver', dest='solver_file',
                        help='Output solver.prototxt file')
    parser.add_argument('--train_test', dest='train_test_file', 
                        help='Output train/test prototxt file')
    parser.add_argument('--tot', dest='layer_tot',
                        help=('Total number of layers.'), type=int,
                        default=50)
    parser.add_argument('--layer', dest='layer_number', nargs='*',
                        help=('Layer number for each layer stage, to use [3,4,6,3] as in resnet50'), type=int,
                        default=[3, 4, 6, 3])
    parser.add_argument('--upd', dest='layer_update', nargs='*',
                        help=('Whether update each layer params, to use [1,1,1,1]'), type=int,
                        default=[1, 1, 1, 1])
    parser.add_argument('--subupd', dest='sublayer_update', 
                        help=('Update from which sublayer, to use z (no update)'), default='z')
    #parser.add_argument('-t', '--type', type=int,
                        #help=('0 for deploy.prototxt, 1 for train_test.prototxt.'),
                        #default=1)
    parser.add_argument('--phase', dest='phase_num',
                        help='0 for train, 1 for test.', type=int,
                        default=1)
    
    args = parser.parse_args()


    if args.layer_tot == 101:
        args.layer_number = [3, 4, 23, 3]

    return args



def generate_data_layer(class_tot, layer_tot, phase='train'):
    data_layer_str = ''
    if phase == 'train':
        data_layer_str = \
'''\
name: "ResNet%d"
layer {
  name: 'input-data'
  type: 'Python'
  top: 'data'
  top: 'im_info'
  top: 'gt_boxes'
  python_param {
    module: 'roi_data_layer.layer'
    layer: 'RoIDataLayer'
    param_str: "'num_classes': %d"
  }
}

'''%(layer_tot, class_tot)

    elif phase == 'test':
        data_layer_str = \
'''\
name: "ResNet%d"
input: "data"
  input_shape {
    dim: 1
    dim: 3
    dim: 224
    dim: 224
  }

  input: "im_info"
  input_shape {
    dim: 1
    dim: 3
  }

'''

    return data_layer_str



def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra", bias_term=False, freeze=False):
    conv_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Convolution"
'''%(layer_name, bottom, top)

    if freeze:
        conv_layer_str += freeze_one_param
        if bias_term:
            conv_layer_str += freeze_one_param

    if bias_term:
        conv_layer_str += '''\
  convolution_param {
    num_output: %d
    kernel_size: %d
    pad: %d
    stride: %d
  }
}

'''%(kernel_num, kernel_size, pad, stride)

    else:
        conv_layer_str += '''\
  convolution_param {
    num_output: %d
    kernel_size: %d
    pad: %d
    stride: %d
    bias_term: false
    weight_filler { type: "%s" std: 0.010 }
  }
}

'''%(kernel_num, kernel_size, pad, stride, filler)

    return conv_layer_str



def generate_pooling_layer(kernel_size, stride, pool_type, layer_name, bottom, top):
    pool_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Pooling"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}

'''%(layer_name, bottom, top, pool_type, kernel_size, stride)
    return pool_layer_str




def generate_fc_layer(num_output, layer_name, bottom, top, filler="msra"):
    fc_layer_str = '''layer {
  bottom: "%s"
  top: "%s"
  name: "%s"
  type: "InnerProduct"
  inner_product_param {
    num_output: %d
  }
}

'''%(bottom, top, layer_name, num_output)
    return fc_layer_str




def generate_eltwise_layer(layer_name, bottom_1, bottom_2, top):
    eltwise_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  type: "Eltwise"
}

'''%(layer_name, bottom_1, bottom_2, top)
    return eltwise_layer_str




def generate_activation_layer(layer_name, bottom, top, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "%s"
}   

'''%(layer_name, bottom, top, act_type)
    return act_layer_str




def generate_softmax_loss(bottom):
    softmax_loss_str = '''layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss"
}

layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
'''%(bottom, bottom)
    return softmax_loss_str




def generate_bn_layer(layer_name_bn, bottom_bn, top_bn, freeze=False):
    bn_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "BatchNorm"
  batch_norm_param {
    use_global_stats: true
  }
'''%(layer_name_bn, bottom_bn, top_bn )

    if freeze:
        bn_layer_str += '''\
  param { lr_mult: 0 }
  param { lr_mult: 0 }
  param { lr_mult: 0 }
'''
    bn_layer_str += '''\
}

layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Scale"
  scale_param {
    bias_term: true
  }
'''%('scale' + layer_name_bn[2:], bottom_bn, top_bn)

    if freeze:
        bn_layer_str += '''\
  param { lr_mult: 0 } 
'''
    
    bn_layer_str += '''\
}

'''

    return bn_layer_str



def generate_rpn_layers(bottom, phase='train'):
    rpn_layers = '''\
#============== RPN ===============
layer {
  name: "rpn_conv/3x3"
  type: "Convolution"
  bottom: "%s"
  top: "rpn/output"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    kernel_size: 3 pad: 1 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_relu/3x3"
  type: "ReLU"
  bottom: "rpn/output"
  top: "rpn/output"
}

layer {
  name: "rpn_cls_score"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_cls_score"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: 18   # 2(bg/fg) * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}

layer {
  name: "rpn_bbox_pred"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_bbox_pred"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: 36   # 4 * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}

layer {
   bottom: "rpn_cls_score"
   top: "rpn_cls_score_reshape"
   name: "rpn_cls_score_reshape"
   type: "Reshape"
   reshape_param { shape { dim: 0 dim: 2 dim: -1 dim: 0 } }
}

'''%bottom
    if phase == 'train':
        rpn_layers += \
'''\
layer {
  name: 'rpn-data'
  type: 'Python'
  bottom: 'rpn_cls_score'
  bottom: 'gt_boxes'
  bottom: 'im_info'
  bottom: 'data'
  top: 'rpn_labels'
  top: 'rpn_bbox_targets'
  top: 'rpn_bbox_inside_weights'
  top: 'rpn_bbox_outside_weights'
  python_param {
    module: 'rpn.anchor_target_layer'
    layer: 'AnchorTargetLayer'
    param_str: "'feat_stride': 16\\n'scales': !!python/tuple [8, 16, 32]"
  }
}

layer {
  name: "rpn_loss_cls"
  type: "SoftmaxWithLoss"
  bottom: "rpn_cls_score_reshape"
  bottom: "rpn_labels"
  propagate_down: 1
  propagate_down: 0
  top: "rpn_cls_loss"
  loss_weight: 1
  loss_param {
    ignore_label: -1
    normalize: true
  }
}

layer {
  name: "rpn_loss_bbox"
  type: "SmoothL1Loss"
  bottom: "rpn_bbox_pred"
  bottom: "rpn_bbox_targets"
  bottom: 'rpn_bbox_inside_weights'
  bottom: 'rpn_bbox_outside_weights'
  top: "rpn_loss_bbox"
  loss_weight: 1
  smooth_l1_loss_param { sigma: 3.0 }
}

'''

    return rpn_layers




def generate_roi_layers(layer_name, bottom, top, class_tot, phase='train'):
    rois_mid_name = '' 
    if phase == 'train':
        rois_mid_name = 'rpn_rois'
    elif phase == 'test':
        rois_mid_name = 'rois'

    roi_layers = '''\
#============== ROI Proposal ===============

layer {
  name: "rpn_cls_prob"
  type: "Softmax"
  bottom: "rpn_cls_score_reshape"
  top: "rpn_cls_prob"
}

layer {
  name: 'rpn_cls_prob_reshape'
  type: 'Reshape'
  bottom: 'rpn_cls_prob'
  top: 'rpn_cls_prob_reshape'
  reshape_param { shape { dim: 0 dim: 18 dim: -1 dim: 0 } }
}

layer {
  name: 'proposal'
  type: 'Python'
  bottom: 'rpn_cls_prob_reshape'
  bottom: 'rpn_bbox_pred'
  bottom: 'im_info'
  top: '%s'
#  top: 'rpn_scores'
  python_param {
    module: 'rpn.proposal_layer'
    layer: 'ProposalLayer'
    param_str: "'feat_stride': 16\\n'scales': !!python/tuple [8, 16, 32]"
  }
}

'''%rois_mid_name

    if phase=='train':
        roi_layers += \
'''\

layer {
  name: 'roi-data'
  type: 'Python'
  bottom: 'rpn_rois'
  bottom: 'gt_boxes'
  top: 'rois'
  top: 'labels'
  top: 'bbox_targets'
  top: 'bbox_inside_weights'
  top: 'bbox_outside_weights'
  python_param {
    module: 'rpn.proposal_target_layer'
    layer: 'ProposalTargetLayer'
    param_str: "'num_classes': %d"
  }
}

'''%class_tot

#if pooled_w = pooled_h = 14, need to use stride 2 in conv5
    roi_layers += \
'''\
layer {
  name: "%s"
  type: "ROIPooling"
  bottom: "%s"
  bottom: "rois"
  top: "%s"
  roi_pooling_param {
    pooled_w: 7
    pooled_h: 7
    spatial_scale: 0.0625 # 1/16
  }
}

'''%(layer_name, bottom, top)

    return roi_layers




def generate_cls_and_bbox_layers(layer_name1, layer_name2, bottom, top1, top2, class_tot, bbox_output_tot):

    cls_and_bbox_layers = '''\
######### Add faster RCNN cls and bbox layer
layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param { lr_mult: 1 decay_mult: 1 }
  param { lr_mult: 2 decay_mult: 0 }
  inner_product_param {
    num_output: %d
    weight_filler { type: "msra" std: 0.010 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param { lr_mult: 1 decay_mult: 1 }
  param { lr_mult: 2 decay_mult: 0 }
  inner_product_param {
    num_output: %d
    weight_filler { type: "msra" std: 0.010 }
    bias_filler { type: "constant" value: 0 }
  }
}
'''%(layer_name1, bottom, top1, class_tot,  layer_name2, bottom, top2, bbox_output_tot)
    return cls_and_bbox_layers





def generate_loss_layers(layer_name1, layer_name2, phase='train'):
    if phase == 'train':
        loss_layers = \
'''\
layer {
  name: "%s"
  type: "SoftmaxWithLoss"
  bottom: "cls_score"
  bottom: "labels"
  propagate_down: 1
  propagate_down: 0
  top: "loss_cls"
  loss_weight: 1
}

layer {
  name: "%s"
  type: "SmoothL1Loss"
  bottom: "bbox_pred"
  bottom: "bbox_targets"
  bottom: "bbox_inside_weights"
  bottom: "bbox_outside_weights"
  top: "loss_bbox"
  loss_weight: 1
}
'''%(layer_name1, layer_name2)
    
    elif phase == 'test':
        loss_layers = \
'''\
layer {
    name: "cls_prob"
    type: "Softmax"
    bottom: "cls_score"
    top: "cls_prob"
}
'''

    return loss_layers





def generate_train_test(args, phase='train'):
    class_tot = 138
    bbox_output_tot = class_tot * 4
    #args = parse_args()
    network_str = generate_data_layer(class_tot, args.layer_tot, phase=phase)
    alphabet = list('abcdefghijklmnopqrstuvwxyz')

    '''stage 1'''
    last_top = 'data'
    #if_freeze = False
    #if_freeze = True
    if_freeze = False if args.layer_update[0] else True
    if_bn_freeze = True
    if args.layer_tot == 50:
        network_str += generate_conv_layer(7, 64, 2, 3, 'conv1', last_top, 'conv1', bias_term=True, freeze=if_freeze)
    else:
        #layer tot 101 or 152
        network_str += generate_conv_layer(7, 64, 2, 3, 'conv1', last_top, 'conv1', bias_term=False, freeze=if_freeze)
    network_str += generate_bn_layer('bn_conv1', 'conv1', 'conv1', freeze=if_bn_freeze)
    network_str += generate_activation_layer('conv1_relu', 'conv1', 'conv1', 'ReLU')
    network_str += generate_pooling_layer(3, 2, 'MAX', 'pool1', 'conv1', 'pool1')

    '''stage 2'''
    last_top = 'pool1'
    #if_freeze = False
    #if_freeze = True
    if_freeze = False if args.layer_update[1] else True
    if_bn_freeze = True
    network_str += generate_conv_layer(1, 256, 1, 0, 'res2a_branch1', last_top, 'res2a_branch1', freeze=if_freeze)
    network_str += generate_bn_layer('bn2a_branch1', 'res2a_branch1', 'res2a_branch1', freeze=if_bn_freeze)
    last_output = 'res2a_branch1'
    for l in alphabet[:args.layer_number[0]]:
        network_str += generate_conv_layer(1, 64, 1, 0, 'res2%s_branch2a'%l, last_top, 'res2%s_branch2a'%l, freeze=if_freeze)
        network_str += generate_bn_layer('bn2%s_branch2a'%l, 'res2%s_branch2a'%l, 'res2%s_branch2a'%l, freeze=if_bn_freeze)
        network_str += generate_activation_layer('res2%s_branch2a_relu'%l, 'res2%s_branch2a'%l, 'res2%s_branch2a'%l, 'ReLU')

        network_str += generate_conv_layer(3, 64, 1, 1, 'res2%s_branch2b'%l, 'res2%s_branch2a'%l, 'res2%s_branch2b'%l, freeze=if_freeze)
        network_str += generate_bn_layer('bn2%s_branch2b'%l, 'res2%s_branch2b'%l, 'res2%s_branch2b'%l, freeze=if_bn_freeze)
        network_str += generate_activation_layer('res2%s_branch2b_relu'%l, 'res2%s_branch2b'%l, 'res2%s_branch2b'%l, 'ReLU')

        network_str += generate_conv_layer(1, 256, 1, 0, 'res2%s_branch2c'%l, 'res2%s_branch2b'%l, 'res2%s_branch2c'%l, freeze=if_freeze)
        network_str += generate_bn_layer('bn2%s_branch2c'%l, 'res2%s_branch2c'%l, 'res2%s_branch2c'%l, freeze=if_bn_freeze)

        network_str += generate_eltwise_layer('res2%s'%l, last_output, 'res2%s_branch2c'%l, 'res2%s'%l)
        network_str += generate_activation_layer('res2%s_relu'%l, 'res2%s'%l, 'res2%s'%l, 'ReLU')

        last_top = 'res2%s'%l
        last_output = 'res2%s'%l

    '''stage 3'''
    #if_freeze = False
    #if_freeze = True
    if_freeze = False if args.layer_update[2] else True
    if_bn_freeze = True
    network_str += generate_conv_layer(1, 512, 2, 0, 'res3a_branch1', last_top, 'res3a_branch1', freeze=if_freeze)
    network_str += generate_bn_layer('bn3a_branch1', 'res3a_branch1', 'res3a_branch1', freeze=if_bn_freeze)
    last_output = 'res3a_branch1'

    for l in alphabet[:args.layer_number[1]]:
        stride = 2 if l == 'a'\
                else 1
        network_str += generate_conv_layer(1, 128, stride, 0, 'res3%s_branch2a'%l, last_top, 'res3%s_branch2a'%l, freeze=if_freeze)
        network_str += generate_bn_layer('bn3%s_branch2a'%l, 'res3%s_branch2a'%l, 'res3%s_branch2a'%l, freeze=if_bn_freeze)
        network_str += generate_activation_layer('res3%s_branch2a_relu'%l, 'res3%s_branch2a'%l, 'res3%s_branch2a'%l, 'ReLU')

        network_str += generate_conv_layer(3, 128, 1, 1, 'res3%s_branch2b'%l, 'res3%s_branch2a'%l, 'res3%s_branch2b'%l, freeze=if_freeze)
        network_str += generate_bn_layer('bn3%s_branch2b'%l, 'res3%s_branch2b'%l, 'res3%s_branch2b'%l, freeze=if_bn_freeze)
        network_str += generate_activation_layer('res3%s_branch2b_relu'%l, 'res3%s_branch2b'%l, 'res3%s_branch2b'%l, 'ReLU')

        network_str += generate_conv_layer(1, 512, 1, 0, 'res3%s_branch2c'%l, 'res3%s_branch2b'%l, 'res3%s_branch2c'%l, freeze=if_freeze)
        network_str += generate_bn_layer('bn3%s_branch2c'%l, 'res3%s_branch2c'%l, 'res3%s_branch2c'%l, freeze=if_bn_freeze)
        network_str += generate_eltwise_layer('res3%s'%l, last_output, 'res3%s_branch2c'%l, 'res3%s'%l)
        network_str += generate_activation_layer('res3%s_relu'%l, 'res3%s'%l, 'res3%s'%l, 'ReLU')
        last_top = 'res3%s'%l
        last_output = 'res3%s'%l

    '''stage 4'''
    #if_freeze = False
    if_freeze = False if args.layer_update[3] else True
    if_bn_freeze = True
    network_str += generate_conv_layer(1, 1024, 2, 0, 'res4a_branch1', last_top, 'res4a_branch1', freeze=if_freeze)
    network_str += generate_bn_layer('bn4a_branch1', 'res4a_branch1', 'res4a_branch1', freeze=if_bn_freeze)
    last_output = 'res4a_branch1'
    for l in alphabet[:args.layer_number[2]]:

        if l >= args.sublayer_update :
            if_freeze = False

        stride = 2 if l == 'a'\
                else 1
        network_str += generate_conv_layer(1, 256, stride, 0, 'res4%s_branch2a'%l, last_top, 'res4%s_branch2a'%l, freeze=if_freeze)
        network_str += generate_bn_layer('bn4%s_branch2a'%l, 'res4%s_branch2a'%l, 'res4%s_branch2a'%l, freeze=if_bn_freeze)
        network_str += generate_activation_layer('res4%s_branch2a_relu'%l, 'res4%s_branch2a'%l, 'res4%s_branch2a'%l, 'ReLU')

        network_str += generate_conv_layer(3, 256, 1, 1, 'res4%s_branch2b'%l, 'res4%s_branch2a'%l, 'res4%s_branch2b'%l, freeze=if_freeze)
        network_str += generate_bn_layer('bn4%s_branch2b'%l, 'res4%s_branch2b'%l, 'res4%s_branch2b'%l, freeze=if_bn_freeze)
        network_str += generate_activation_layer('res4%s_branch2b_relu'%l, 'res4%s_branch2b'%l, 'res4%s_branch2b'%l, 'ReLU')

        network_str += generate_conv_layer(1, 1024, 1, 0, 'res4%s_branch2c'%l, 'res4%s_branch2b'%l, 'res4%s_branch2c'%l, freeze=if_freeze)
        network_str += generate_bn_layer('bn4%s_branch2c'%l, 'res4%s_branch2c'%l, 'res4%s_branch2c'%l, freeze=if_bn_freeze)
        network_str += generate_eltwise_layer('res4%s'%l, last_output, 'res4%s_branch2c'%l, 'res4%s'%l)
        network_str += generate_activation_layer('res4%s_relu'%l, 'res4%s'%l, 'res4%s'%l, 'ReLU')
        last_top = 'res4%s'%l
        last_output = 'res4%s'%l
    
    '''RPN and ROI'''
    network_str += generate_rpn_layers(last_top, phase=phase)    
    network_str += generate_roi_layers('roi_pool5', last_output, 'roi_pool5', class_tot, phase=phase)
    last_top = 'roi_pool5'

    '''stage 5'''
    #if_freeze = False
    if_bn_freeze = True
    #network_str += generate_conv_layer(1, 2048, 2, 0, 'res5a_branch1', last_top, 'res5a_branch1')
    network_str += generate_conv_layer(1, 2048, 1, 0, 'res5a_branch1', last_top, 'res5a_branch1')
    network_str += generate_bn_layer('bn5a_branch1', 'res5a_branch1', 'res5a_branch1', freeze=if_bn_freeze)
    last_output = 'res5a_branch1'
    for l in alphabet[:args.layer_number[3]]:
        #stride = 2 if l == 'a'\
                #else 1
        stride = 1
        network_str += generate_conv_layer(1, 512, stride, 0, 'res5%s_branch2a'%l, last_top, 'res5%s_branch2a'%l)
        network_str += generate_bn_layer('bn5%s_branch2a'%l, 'res5%s_branch2a'%l, 'res5%s_branch2a'%l, freeze=if_bn_freeze)
        network_str += generate_activation_layer('res5%s_branch2a_relu'%l, 'res5%s_branch2a'%l, 'res5%s_branch2a'%l, 'ReLU')
        
        network_str += generate_conv_layer(3, 512, 1, 1, 'res5%s_branch2b'%l, 'res5%s_branch2a'%l, 'res5%s_branch2b'%l)
        network_str += generate_bn_layer('bn5%s_branch2b'%l, 'res5%s_branch2b'%l, 'res5%s_branch2b'%l, freeze=if_bn_freeze)
        network_str += generate_activation_layer('res5%s_branch2b_relu'%l, 'res5%s_branch2b'%l, 'res5%s_branch2b'%l, 'ReLU')

        network_str += generate_conv_layer(1, 2048, 1, 0, 'res5%s_branch2c'%l, 'res5%s_branch2b'%l, 'res5%s_branch2c'%l)
        network_str += generate_bn_layer('bn5%s_branch2c'%l, 'res5%s_branch2c'%l, 'res5%s_branch2c'%l, freeze=if_bn_freeze)
        network_str += generate_eltwise_layer('res5%s'%l, last_output, 'res5%s_branch2c'%l, 'res5%s'%l)
        network_str += generate_activation_layer('res5%s_relu'%l, 'res5%s'%l, 'res5%s'%l, 'ReLU')
        last_top = 'res5%s'%l
        last_output = 'res5%s'%l

    network_str += generate_pooling_layer(7, 1, 'AVE', 'pool5', last_top, 'pool5')
    #network_str += generate_fc_layer(3, 'fc3', 'pool5', 'fc3', 'gaussian')
    #network_str += generate_softmax_loss('fc3')
    
    network_str += generate_cls_and_bbox_layers('cls_score', 'bbox_pred', 'pool5', 'cls_score', 'bbox_pred', class_tot, bbox_output_tot)
    network_str += generate_loss_layers('loss_cls', 'loss_bbox', phase=phase)


    return network_str




def generate_solver(train_name, layer_tot):
    solver_str = \
'''\
train_net: "%s"
base_lr: 0.001
lr_policy: "step"
gamma: 0.1
stepsize: 50000
display: 20
average_loss: 100
momentum: 0.9
weight_decay: 0.0001
snapshot: 0
snapshot_prefix: "resnet%d_faster_rcnn"
iter_size: 2'''%(train_name, layer_tot)

    return solver_str




def main():
    #usage: 
    #python tools/resnet_faster-rcnn_generator.py --tot 50 --train_test  models/RESNET/faster_rcnn_end2end_houzzdata1/train50_frz-conv3.prototxt --phase 0 --upd 0 0 0 1
    #python tools/resnet_faster-rcnn_generator.py --tot 101 --train_test  models/RESNET/faster_rcnn_end2end_houzzdata1/train101_frz-conv4p.prototxt --phase 0 --upd 0 0 0 0 --subupd q
    #python tools/resnet_faster-rcnn_generator.py --tot 101 --train_test  models/RESNET/faster_rcnn_end2end_houzzdata1/train101_frz-conv3-new.prototxt --phase 0 --upd 0 0 0 1
    #python tools/resnet_faster-rcnn_generator.py --tot 101 --train_test  models/RESNET/faster_rcnn_end2end_houzzdata1/test101.prototxt --phase 1 --upd 0 0 0 0
    #python tools/resnet_faster-rcnn_generator.py --tot 101 --train_test  models/RESNET/faster_rcnn_end2end_houzzdata1/train101_frz-conv4p-new.prototxt --phase 0 --upd 0 0 0 0 --subupd q
    #python tools/resnet_faster-rcnn_generator.py --tot 101 --train_test  models/RESNET/faster_rcnn_end2end_houzzdata1/train101_frz-conv4f-new.prototxt --phase 0 --upd 0 0 0 0 --subupd g
    #python tools/resnet_faster-rcnn_generator.py --tot 101 --train_test  models/RESNET/faster_rcnn_end2end_houzzdata1/train101_frz-conv4v-new.prototxt --phase 0 --upd 0 0 0 0 --subupd w
    #python tools/resnet_faster-rcnn_generator.py --tot 50 --train_test  models/RESNET/faster_rcnn_end2end_houzzdata1/train50_frz-conv3-new.prototxt --phase 0 --upd 0 0 0 1

    #python tools/resnet_faster-rcnn_generator_v8.py --tot 50 --train_test  models/houzzdata/RESNET/faster_rcnn_end2end_houzzdata1/train50_v8_gen_frz-conv3.prototxt --phase 0 --upd 0 0 0 1

    args = parse_args()
    phase = ''
    if args.phase_num == 0:
        phase = 'train'
    elif args.phase_num == 1:
        phase = 'test'
     
    assert phase == 'train' or phase == 'test', "Phase is not properly specified."

    #print "args"
    #embed()

    if args.train_test_file is not None:
        network_str = generate_train_test(args, phase)
        fp = open(args.train_test_file, 'w')
        fp.write(network_str)
        fp.close()
    if args.solver_file is not None:
        solver_str = generate_solver(args.train_test_file, args.layer_tot)
        fp = open(args.solver_file, 'w')
        fp.write(solver_str)
        fp.close()



if __name__ == '__main__':
    main()
