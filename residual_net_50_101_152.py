"""
@author: Yuhang He 
@Date: Mar. 26, 2016
@Email: yuhanghe@whu.edu.cn
@ReadMe: script generate_residual_net.py and residual_net.py are utilized to create residual net 
         50/101/152 layer.
@Paper for Reference: Identity Mapping in Deep Residual Networks by Kaiming He et al.
"""

import caffe
from caffe import layers as L
from caffe import params as P


def bn_scale_relu( bottom ):
    bn = L.BatchNorm( bottom, use_global_stats=False )
    bn_scale = L.Scale( bn, scale_param = dict( bias_term=True), in_place = True )
    bn_scale_relu = L.ReLU( bn_scale, in_place = True )

    return bn, bn_scale, bn_scale_relu

def conv( bottom, num_output = 64, kernel_size = 3, stride = 1, pad = 0 ):
    conv_layer = L.Convolution( bottom, num_output = num_output, kernel_size = kernel_size, \
                          stride = stride, pad = pad, \
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                          weight_filler=dict(type='xavier', std=0.01),\
                          bias_filler=dict(type='constant', value=0) )
    return conv_layer

def fc_relu_drop(bottom, num_output=1024, dropout_ratio=0.5):
    fc = L.InnerProduct(bottom, num_output=num_output,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='xavier', std=1),
                        bias_filler=dict(type='constant', value=0.2))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, relu, drop


def block_conv_bn_scale_relu( bottom, num_output = 64, kernel_size = 3, stride = 1, pad = 0 ):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    conv_bn = L.BatchNorm( conv, use_global_stats = False, in_place = True )
    conv_scale = L.Scale( conv, scale_param = dict( bias_term = True ), in_place = True )
    conv_relu = L.ReLU( conv, in_place = True )

    return conv, conv_bn, conv_scale, conv_relu


def block_conv_bn_scale(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0.2))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)

    return conv, conv_bn, conv_scale


def eletwise( bottom1, bottom2 ):
    eletwise_layer = L.Eltwise( bottom1, bottom2, eltwise_param = dict( operation = 1) )
    
    return eletwise_layer


def redisual_connect_no_dimen_match( bottom, base_output = 64 ):
    res_i_bn, res_i_scale, res_i_relu = bn_scale_relu( bottom )
    res_i_conv1, res_i_conv1_bn, res_i_conv1_scale, res_i_conv1_relu = \
        block_conv_bn_scale_relu( res_i_relu, num_output = base_output, kernel_size = 1, stride = 1, pad = 0 )
    res_i_conv2, res_i_conv2_bn, res_i_conv2_scale, res_i_conv2_relu = \
        block_conv_bn_scale_relu( res_i_conv1_relu, num_output = base_output, kernel_size = 3, pad = 1 )
    res_i_conv3 = \
        conv( res_i_conv2_relu, num_output = 4*base_output, kernel_size = 1, stride = 1, pad = 0 )

    eletwise_layer = eletwise( bottom, res_i_conv3 )
    
    return res_i_bn, res_i_scale, res_i_relu, res_i_conv1, res_i_conv1_bn, res_i_conv1_scale, res_i_conv1_relu, res_i_conv2, res_i_conv2_bn, res_i_conv2_scale, res_i_conv2_relu, res_i_conv3, eletwise_layer


def redisual_connect_with_dimen_match( bottom, stride = 1, base_output = 64 ):
    res_i_bn, res_i_scale, res_i_relu = bn_scale_relu( bottom )
    res_i_conv1, res_i_conv1_bn, res_i_conv1_scale, res_i_conv1_relu = block_conv_bn_scale_relu( res_i_relu, num_output = base_output, kernel_size = 1, stride = stride + 1, pad = 0 )
    res_i_conv2, res_i_conv2_bn, res_i_conv2_scale, res_i_conv2_relu = block_conv_bn_scale_relu( res_i_conv1_relu, num_output = base_output, kernel_size = 3, pad = 1 )
    res_i_conv3 = conv( res_i_conv2_relu, num_output = 4*base_output, kernel_size = 1, stride = 1, pad = 0 )

    #in the dimension match layer, the stride should be 2
    res_i_match_conv, res_i_match_bn, res_i_match_scale = block_conv_bn_scale( bottom, num_output = 4*base_output, kernel_size = 1, stride = 2, pad = 0 )

    eletwise_layer = eletwise( res_i_match_scale, res_i_conv3 )
    return res_i_bn, res_i_scale, res_i_relu, res_i_conv1, res_i_conv1_bn, res_i_conv1_scale, res_i_conv1_relu, res_i_conv2, res_i_conv2_bn, res_i_conv2_scale, res_i_conv2_relu, res_i_conv3, res_i_match_conv, res_i_match_bn, res_i_match_scale, eletwise_layer


def redisual_connect_with_dimen_match_no_patch_reduce( bottom, stride = 1, base_output = 64 ):
    res_i_bn, res_i_scale, res_i_relu = bn_scale_relu( bottom )
    res_i_conv1, res_i_conv1_bn, res_i_conv1_scale, res_i_conv1_relu = block_conv_bn_scale_relu( res_i_relu, num_output = base_output, kernel_size = 1, stride = stride, pad = 0 )
    res_i_conv2, res_i_conv2_bn, res_i_conv2_scale, res_i_conv2_relu = block_conv_bn_scale_relu( res_i_conv1_relu, num_output = base_output, kernel_size = 3, pad = 1 )
    res_i_conv3 = conv( res_i_conv2_relu, num_output = 4*base_output, kernel_size = 1, stride = 1, pad = 0 )

    #in the dimension match layer, the stride should be 2
    res_i_match_conv, res_i_match_bn, res_i_match_scale = block_conv_bn_scale( bottom, num_output = 4*base_output, kernel_size = 1, stride = 1, pad = 0 )

    eletwise_layer = eletwise( res_i_match_scale, res_i_conv3 )
    return res_i_bn, res_i_scale, res_i_relu, res_i_conv1, res_i_conv1_bn, res_i_conv1_scale, res_i_conv1_relu, res_i_conv2, res_i_conv2_bn, res_i_conv2_scale, res_i_conv2_relu, res_i_conv3, res_i_match_conv, res_i_match_bn, res_i_match_scale, eletwise_layer




skip_connect_no_dimen_match = 'n.res(stage)_bn, n_res(stage)_scale, n.res(stage)_relu, \
n.res(stage)_conv1, n.res(stage)_conv1_bn, n.res(stage)_conv1_scale, n.res(stage)_conv1_relu,\
n.res(stage)_conv2, n.res(stage)_conv2_bn, n.res(stage)_conv2_scale, n.res(stage)_conv2_relu,\
n.res(stage)_conv3, n.res(stage)_eletwise = \
redisual_connect_no_dimen_match( (bottom), base_output = (num) )'

skip_connect_with_dimen_match = 'n.res(stage)_bn, n.res(stage)_scale, n.res(stage)_relu, \
n.res(stage)_conv1, n.res(stage)_conv1_bn, n.res(stage)_conv1_scale, n.res(stage)_conv1_relu,\
n.res(stage)_conv2, n.res(stage)_conv2_bn, n.res(stage)_conv2_scale, n.res(stage)_conv2_relu,\
n.res(stage)_conv3, n.res(stage)_match_conv, n.res(stage)_match_bn, n.res(stage)_match_scale, n.res(stage)_eletwise = \
redisual_connect_with_dimen_match( (bottom), base_output = (num) )'

skip_connect_with_dimen_match_no_patch_reduce = 'n.res(stage)_bn, n.res(stage)_scale, n.res(stage)_relu, \
n.res(stage)_conv1, n.res(stage)_conv1_bn, n.res(stage)_conv1_scale, n.res(stage)_conv1_relu,\
n.res(stage)_conv2, n.res(stage)_conv2_bn, n.res(stage)_conv2_scale, n.res(stage)_conv2_relu,\
n.res(stage)_conv3, n.res(stage)_match_conv, n.res(stage)_match_bn, n.res(stage)_match_scale, n.res(stage)_eletwise = \
redisual_connect_with_dimen_match_no_patch_reduce( (bottom), base_output = (num) )'


class ResNet(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def resnet_layers_proto( self, batch_size, phase = 'TRAIN', stages = (3, 4, 6, 3) ):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            need_mirror = True
        else:
            source_data = self.test_data
            need_mirror = False
        n.data, n.label = L.Data( source = source_data, backend = P.Data.LMDB, batch_size = batch_size, ntop =2,
                                 transform_param=dict( crop_size=224, mean_value=[128, 128, 128], mirror = need_mirror))

        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = \
            block_conv_bn_scale_relu( n.data, num_output = 64, kernel_size = 7, stride = 2, pad = 3 )  # 64x112x112
        n.pool1 = L.Pooling( n.conv1, kernel_size = 3, stride = 2, pool = P.Pooling.MAX ) 
       
        residual_num = 0 
        for num in xrange(len(stages)): 
            for i in xrange(stages[num]):
                residual_num = residual_num + 1
               
                if num == 0 and i == 0:
                  stage_string = skip_connect_with_dimen_match_no_patch_reduce
                  if residual_num == 1:
                    bottom_string = 'n.pool1'
                  else:
                    bottom_string = 'n.res%s_eletwise'%( str( residual_num -1 ) )
                elif i == 0 and num > 0:
                    stage_string = skip_connect_with_dimen_match
                    if residual_num == 1:
                        bottom_string = 'n.pool1'
                    else:
                        bottom_string = 'n.res%s_eletwise'%( str( residual_num -1 ) )
                else:
                    stage_string = skip_connect_no_dimen_match
                    bottom_string = 'n.res%s_eletwise'%( str( residual_num -1 ) )
                exec (stage_string.replace('(stage)', str(residual_num)).replace('(bottom)', bottom_string).replace('(num)', str( 2 ** num * 64 )))
        
        exec 'n.pool5 = L.Pooling( bottom_string, kernel_size=7, stride=1, pool=P.Pooling.AVE)'.replace('bottom_string', 'n.res%s_eletwise'%str( residual_num ))
        
        n.classifier = L.InnerProduct(n.pool5, num_output=self.classifier_num,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))
        n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1 = L.Accuracy(n.classifier, n.label, include=dict(phase=1))
            n.accuracy_top5 = L.Accuracy(n.classifier, n.label, include=dict(phase=1),
                                       accuracy_param=dict(top_k=5))


        return n.to_proto()
