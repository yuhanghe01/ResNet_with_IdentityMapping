"""
@Author: Yuhang He
@Date: Mar. 27, 2016
@Modified: Mar. 31, 2016. added 18/34 layer generation script
@Email: yuhanghe@whu.edu.cn
"""
CAFFE='/home/heyuhang/caffe_latest/caffe_jitter/python'
import sys
sys.path.append(CAFFE)

from caffe.proto import caffe_pb2
import residual_net_18_34
import residual_net_50_101_152


def save_proto(proto, prototxt):
    with open(prototxt, 'w') as f:
        f.write(str(proto))

layer_18_layer = (2, 2, 2, 2)
layer_34_layer = (3, 4, 6, 3)

layer_50_layer = layer_34_layer
layer_101_layer = (3, 4, 23, 3)
layer_152_layer = (3, 8, 36, 3)

if __name__ == '__main__':
    layer_to_generate = layer_18_layer
    output_num = 7
    model = residual_net_18_34.ResNet('imagenet_test_lmdb', 'imagenet_train_lmdb', output_num )
    train_proto = model.resnet_layers_proto( batch_size = 64, phase = "TRAIN", stages = layer_to_generate )
    test_proto = model.resnet_layers_proto( batch_size = 50, phase="TEST", stages = layer_to_generate )
    save_proto(train_proto, '18_layer_train_identity_mapping.prototxt')
    save_proto(test_proto, '18_layer_test_identity_mapping.prototxt')

    layer_to_generate = layer_50_layer
    output_num = 7
    model = residual_net_50_101_152.ResNet('imagenet_test_lmdb', 'imagenet_train_lmdb', output_num )
    train_proto = model.resnet_layers_proto( batch_size = 64, phase = "TRAIN", stages = layer_to_generate )
    test_proto = model.resnet_layers_proto( batch_size = 50, phase="TEST", stages = layer_to_generate )
    save_proto(train_proto, '50_layer_train_identity_mapping.prototxt')
    save_proto(test_proto, '50_layer_test_identity_mapping.prototxt')
