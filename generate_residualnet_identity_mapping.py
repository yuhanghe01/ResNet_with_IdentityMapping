"""
@Author: Yuhang He
@Date: Mar. 27, 2016
@Email: yuhanghe@whu.edu.cn
"""



CAFFE = '/home/heyuhang/caffe_latest/caffe_jitter/python'
import sys
sys.path.append(CAFFE)

from caffe.proto import caffe_pb2
import inception_v3
import residual_net

def save_proto(proto, prototxt):
    with open(prototxt, 'w') as f:
        f.write(str(proto))



layer_50_layer = (3, 4, 6, 3)
layer_101_layer = (3, 4,  23, 3)
layer_152_layer = (3, 8, 38, 3)

if __name__ == '__main__':
    layer_to_generate = layer_50_layer
    output_num = 7
    model = residual_net.ResNet('imagenet_test_lmdb', 'imagenet_train_lmdb', output_num )
    train_proto = model.resnet_layers_proto( batch_size = 64, phase = "TRAIN", stages = layer_to_generate )
    test_proto = model.resnet_layers_proto( batch_size = 50, phase="TEST", stages = layer_to_generate )

    save_proto(train_proto, 'test_imagenet_train.prototxt')
    save_proto(test_proto, 'test_imagenet_test.prototxt')
