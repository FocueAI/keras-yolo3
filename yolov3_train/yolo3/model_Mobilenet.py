"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import merge,Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from keras.regularizers import l2

from yolo3.utils import compose
import math


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def leakyRelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    # 416 x 416 x 3
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)

    # 208 x 208 x 32
    x = resblock_body(x, 64, 1)

    # 208 x 208 x 64
    x = resblock_body(x, 128, 2)

    # 104 x 104 x 128
    x = resblock_body(x, 256, 8)

    # 52 x 52 x 256
    x = resblock_body(x, 512, 8)

    # 26 x 26 x 512
    x = resblock_body(x, 1024, 4)

    # 13 x 13 x 1024
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras.  这是yolo的主干网络"""
    '''Layer Nanem: input_1 Output: Tensor("input_1:0", shape=(?, 416, 416, 3), dtype=float32)
    Layer Nanem: conv1_pad Output: Tensor("conv1_pad/Pad:0", shape=(?, 418, 418, 3), dtype=float32)
    Layer Nanem: conv1 Output: Tensor("conv1/convolution:0", shape=(?, 208, 208, 32), dtype=float32)
    Layer Nanem: conv1_bn Output: Tensor("conv1_bn/cond/Merge:0", shape=(?, 208, 208, 32), dtype=float32)
    Layer Nanem: conv1_relu Output: Tensor("conv1_relu/Minimum:0", shape=(?, 208, 208, 32), dtype=float32)
    Layer Nanem: conv_pad_1 Output: Tensor("conv_pad_1/Pad:0", shape=(?, 210, 210, 32), dtype=float32)
    Layer Nanem: conv_dw_1 Output: Tensor("conv_dw_1/depthwise:0", shape=(?, 208, 208, 32), dtype=float32)
    Layer Nanem: conv_dw_1_bn Output: Tensor("conv_dw_1_bn/cond/Merge:0", shape=(?, 208, 208, 32), dtype=float32)
    Layer Nanem: conv_dw_1_relu Output: Tensor("conv_dw_1_relu/Minimum:0", shape=(?, 208, 208, 32), dtype=float32)
    Layer Nanem: conv_pw_1 Output: Tensor("conv_pw_1/convolution:0", shape=(?, 208, 208, 64), dtype=float32)
    Layer Nanem: conv_pw_1_bn Output: Tensor("conv_pw_1_bn/cond/Merge:0", shape=(?, 208, 208, 64), dtype=float32)
    Layer Nanem: conv_pw_1_relu Output: Tensor("conv_pw_1_relu/Minimum:0", shape=(?, 208, 208, 64), dtype=float32)
    Layer Nanem: conv_pad_2 Output: Tensor("conv_pad_2/Pad:0", shape=(?, 210, 210, 64), dtype=float32)
    Layer Nanem: conv_dw_2 Output: Tensor("conv_dw_2/depthwise:0", shape=(?, 104, 104, 64), dtype=float32)
    Layer Nanem: conv_dw_2_bn Output: Tensor("conv_dw_2_bn/cond/Merge:0", shape=(?, 104, 104, 64), dtype=float32)
    Layer Nanem: conv_dw_2_relu Output: Tensor("conv_dw_2_relu/Minimum:0", shape=(?, 104, 104, 64), dtype=float32)
    Layer Nanem: conv_pw_2 Output: Tensor("conv_pw_2/convolution:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Nanem: conv_pw_2_bn Output: Tensor("conv_pw_2_bn/cond/Merge:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Nanem: conv_pw_2_relu Output: Tensor("conv_pw_2_relu/Minimum:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Nanem: conv_pad_3 Output: Tensor("conv_pad_3/Pad:0", shape=(?, 106, 106, 128), dtype=float32)
    Layer Nanem: conv_dw_3 Output: Tensor("conv_dw_3/depthwise:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Nanem: conv_dw_3_bn Output: Tensor("conv_dw_3_bn/cond/Merge:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Nanem: conv_dw_3_relu Output: Tensor("conv_dw_3_relu/Minimum:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Nanem: conv_pw_3 Output: Tensor("conv_pw_3/convolution:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Nanem: conv_pw_3_bn Output: Tensor("conv_pw_3_bn/cond/Merge:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Nanem: conv_pw_3_relu Output: Tensor("conv_pw_3_relu/Minimum:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Nanem: conv_pad_4 Output: Tensor("conv_pad_4/Pad:0", shape=(?, 106, 106, 128), dtype=float32)
    Layer Nanem: conv_dw_4 Output: Tensor("conv_dw_4/depthwise:0", shape=(?, 52, 52, 128), dtype=float32)
    Layer Nanem: conv_dw_4_bn Output: Tensor("conv_dw_4_bn/cond/Merge:0", shape=(?, 52, 52, 128), dtype=float32)
    Layer Nanem: conv_dw_4_relu Output: Tensor("conv_dw_4_relu/Minimum:0", shape=(?, 52, 52, 128), dtype=float32)
    Layer Nanem: conv_pw_4 Output: Tensor("conv_pw_4/convolution:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Nanem: conv_pw_4_bn Output: Tensor("conv_pw_4_bn/cond/Merge:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Nanem: conv_pw_4_relu Output: Tensor("conv_pw_4_relu/Minimum:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Nanem: conv_pad_5 Output: Tensor("conv_pad_5/Pad:0", shape=(?, 54, 54, 256), dtype=float32)
    Layer Nanem: conv_dw_5 Output: Tensor("conv_dw_5/depthwise:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Nanem: conv_dw_5_bn Output: Tensor("conv_dw_5_bn/cond/Merge:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Nanem: conv_dw_5_relu Output: Tensor("conv_dw_5_relu/Minimum:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Nanem: conv_pw_5 Output: Tensor("conv_pw_5/convolution:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Nanem: conv_pw_5_bn Output: Tensor("conv_pw_5_bn/cond/Merge:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Nanem: conv_pw_5_relu Output: Tensor("conv_pw_5_relu/Minimum:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Nanem: conv_pad_6 Output: Tensor("conv_pad_6/Pad:0", shape=(?, 54, 54, 256), dtype=float32)
    Layer Nanem: conv_dw_6 Output: Tensor("conv_dw_6/depthwise:0", shape=(?, 26, 26, 256), dtype=float32)
    Layer Nanem: conv_dw_6_bn Output: Tensor("conv_dw_6_bn/cond/Merge:0", shape=(?, 26, 26, 256), dtype=float32)
    Layer Nanem: conv_dw_6_relu Output: Tensor("conv_dw_6_relu/Minimum:0", shape=(?, 26, 26, 256), dtype=float32)
    Layer Nanem: conv_pw_6 Output: Tensor("conv_pw_6/convolution:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_6_bn Output: Tensor("conv_pw_6_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_6_relu Output: Tensor("conv_pw_6_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pad_7 Output: Tensor("conv_pad_7/Pad:0", shape=(?, 28, 28, 512), dtype=float32)
    Layer Nanem: conv_dw_7 Output: Tensor("conv_dw_7/depthwise:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_7_bn Output: Tensor("conv_dw_7_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_7_relu Output: Tensor("conv_dw_7_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_7 Output: Tensor("conv_pw_7/convolution:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_7_bn Output: Tensor("conv_pw_7_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_7_relu Output: Tensor("conv_pw_7_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pad_8 Output: Tensor("conv_pad_8/Pad:0", shape=(?, 28, 28, 512), dtype=float32)
    Layer Nanem: conv_dw_8 Output: Tensor("conv_dw_8/depthwise:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_8_bn Output: Tensor("conv_dw_8_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_8_relu Output: Tensor("conv_dw_8_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_8 Output: Tensor("conv_pw_8/convolution:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_8_bn Output: Tensor("conv_pw_8_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_8_relu Output: Tensor("conv_pw_8_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pad_9 Output: Tensor("conv_pad_9/Pad:0", shape=(?, 28, 28, 512), dtype=float32)
    Layer Nanem: conv_dw_9 Output: Tensor("conv_dw_9/depthwise:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_9_bn Output: Tensor("conv_dw_9_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_9_relu Output: Tensor("conv_dw_9_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_9 Output: Tensor("conv_pw_9/convolution:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_9_bn Output: Tensor("conv_pw_9_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_9_relu Output: Tensor("conv_pw_9_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pad_10 Output: Tensor("conv_pad_10/Pad:0", shape=(?, 28, 28, 512), dtype=float32)
    Layer Nanem: conv_dw_10 Output: Tensor("conv_dw_10/depthwise:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_10_bn Output: Tensor("conv_dw_10_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_10_relu Output: Tensor("conv_dw_10_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_10 Output: Tensor("conv_pw_10/convolution:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_10_bn Output: Tensor("conv_pw_10_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_10_relu Output: Tensor("conv_pw_10_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pad_11 Output: Tensor("conv_pad_11/Pad:0", shape=(?, 28, 28, 512), dtype=float32)
    Layer Nanem: conv_dw_11 Output: Tensor("conv_dw_11/depthwise:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_11_bn Output: Tensor("conv_dw_11_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_dw_11_relu Output: Tensor("conv_dw_11_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_11 Output: Tensor("conv_pw_11/convolution:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_11_bn Output: Tensor("conv_pw_11_bn/cond/Merge:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pw_11_relu Output: Tensor("conv_pw_11_relu/Minimum:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Nanem: conv_pad_12 Output: Tensor("conv_pad_12/Pad:0", shape=(?, 28, 28, 512), dtype=float32)
    Layer Nanem: conv_dw_12 Output: Tensor("conv_dw_12/depthwise:0", shape=(?, 13, 13, 512), dtype=float32)
    Layer Nanem: conv_dw_12_bn Output: Tensor("conv_dw_12_bn/cond/Merge:0", shape=(?, 13, 13, 512), dtype=float32)
    Layer Nanem: conv_dw_12_relu Output: Tensor("conv_dw_12_relu/Minimum:0", shape=(?, 13, 13, 512), dtype=float32)
    Layer Nanem: conv_pw_12 Output: Tensor("conv_pw_12/convolution:0", shape=(?, 13, 13, 1024), dtype=float32)
    Layer Nanem: conv_pw_12_bn Output: Tensor("conv_pw_12_bn/cond/Merge:0", shape=(?, 13, 13, 1024), dtype=float32)
    Layer Nanem: conv_pw_12_relu Output: Tensor("conv_pw_12_relu/Minimum:0", shape=(?, 13, 13, 1024), dtype=float32)
    Layer Nanem: conv_pad_13 Output: Tensor("conv_pad_13/Pad:0", shape=(?, 15, 15, 1024), dtype=float32)
    Layer Nanem: conv_dw_13 Output: Tensor("conv_dw_13/depthwise:0", shape=(?, 13, 13, 1024), dtype=float32)
    Layer Nanem: conv_dw_13_bn Output: Tensor("conv_dw_13_bn/cond/Merge:0", shape=(?, 13, 13, 1024), dtype=float32)
    Layer Nanem: conv_dw_13_relu Output: Tensor("conv_dw_13_relu/Minimum:0", shape=(?, 13, 13, 1024), dtype=float32)
    Layer Nanem: conv_pw_13 Output: Tensor("conv_pw_13/convolution:0", shape=(?, 13, 13, 1024), dtype=float32)
    Layer Nanem: conv_pw_13_bn Output: Tensor("conv_pw_13_bn/cond/Merge:0", shape=(?, 13, 13, 1024), dtype=float32)
    Layer Nanem: conv_pw_13_relu Output: Tensor("conv_pw_13_relu/Minimum:0", shape=(?, 13, 13, 1024), dtype=float32)
    Layer Nanem: global_average_pooling2d_1 Output: Tensor("global_average_pooling2d_1/Mean:0", shape=(?, 1024), dtype=float32)
    Layer Nanem: reshape_1 Output: Tensor("reshape_1/Reshape:0", shape=(?, 1, 1, 1024), dtype=float32)
    Layer Nanem: dropout Output: Tensor("dropout/cond/Merge:0", shape=(?, 1, 1, 1024), dtype=float32)
    Layer Nanem: conv_preds Output: Tensor("conv_preds/BiasAdd:0", shape=(?, 1, 1, 1000), dtype=float32)
    Layer Nanem: act_softmax Output: Tensor("act_softmax/truediv:0", shape=(?, 1, 1, 1000), dtype=float32)
    Layer Nanem: reshape_2 Output: Tensor("reshape_2/Reshape:0", shape=(?, 1000), dtype=float32)
    '''

    #net, endpoint = inception_v2.inception_v2(inputs)
    mobilenet = MobileNet(input_tensor=inputs,weights='imagenet')
    # mobilenet = MobileNet(input_tensor=(416,416,3), weights='imagenet')
    mobilenet.summary()

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256

    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f1 :13 x 13 x 1024
    x, y1 = make_last_layers(f1, 512, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = mobilenet.get_layer('conv_pw_11_relu').output
    # f2: 26 x 26 x 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = mobilenet.get_layer('conv_pw_5_relu').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
    # 27 = 类别数 + (x0,y0,w,h.confidence)
    return Model(inputs = inputs, outputs=[y1,y2,y3])  # [ (?,?,?,27),(?,?,?,27),(?,?,?,27) ] ===>

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

# 自己实现的方法
# def yolo_head(feature_map,anchors,num_cls,input_shape,calc_loss=False):
# # def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
#     """ 用于对yolo_body提取出来的特征进行特征梳理,以及一些编码操作"""
#     batch_size = feature_map.shape[0]
#     # size_x,size_y = feature_map.get_shape()[1].value,feature_map.get_shape()[2].value
#     # size_x, size_y = feature_map.get_shape()[1].value, feature_map.get_shape()[2]
#     size_x, size_y = feature_map.shape[1].value, feature_map.shape[2].value
#     # size_x, size_y = int(feature_map.shape[1]), int(feature_map.shape[2])
#     print('feature_map:',feature_map)
#     print('size_x:',size_x)
#     anchors_tensor = tf.cast(tf.convert_to_tensor(anchors),K.dtype(feature_map)) # shape = [3,2]
#
#     grid_x = K.reshape(K.tile(K.arange(start=0, stop=size_x), size_y),(size_y,size_x)) # [52,52]
#     grid_x = K.expand_dims(grid_x,axis=-1) # [52,52,1]
#
#     # grid_y = K.tile(K.arange(start=0, stop=size_y), (size_x, 1)).T # [52,52]
#     grid_y = K.reshape(K.tile(K.arange(start=0, stop=size_y), size_x), (size_x, size_y))  # [52,52]
#     grid_y = K.expand_dims(grid_y, axis=-1)  # [52,52,1]
#
#     grid_xy = K.concatenate((grid_x,grid_y),axis=-1) # [52,52,2]
#     grid_xy = tf.cast(grid_xy,K.dtype(feature_map))
#     # 为了后面 pre_centers 的计算 广播机制
#     grid_xy = K.expand_dims(grid_xy,axis=-2)
#
#
#
#     feature_map = K.reshape(feature_map,(-1,size_x,size_y,3,5+num_cls))
#     #            shape=        [?,52,52,3,2]
#     pre_centers = ( K.sigmoid(feature_map[...,0:2]) +  grid_xy) /tf.cast(K.shape(grid_xy)[0:2],K.dtype(feature_map)) # 0~1之间的相对值
#     #            [3,2]         [?,52,52,3,2]
#     # print("K.shape(input_shape)[1:3]:",input_shape)
#     pre_wh = anchors_tensor * K.exp(feature_map[...,2:4])/tf.cast(input_shape, K.dtype(feature_map))
#     # feature_map = K.reshape(feature_map,(-1,grid_xy))
#     pre_confidence = feature_map[...,4:5]
#     pre_cls = feature_map[...,5:]
#
#     print('pre_centers:',pre_centers)
#     print('pre_wh:', pre_wh)
#     print('grid_xy:',grid_xy)  # shape=[?,52,52,30]  30 = 3*(5+5个类别)
#     return grid_xy,feature_map,pre_centers,pre_wh

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters.
       feats : 网络预测的某一个feature map 上的anchor大小参数
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.    输入进来的 feats shape [-1,13,13,num_anchors*(num_classes+5)]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])  # anchor

    grid_shape = K.shape(feats)[1:3]    # 该层网络输出的feature的 height, width  shape = (2,)
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])                                               # [ grid_shape[0],grid_shape[1],1,1 ]
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), # [ grid_shape[0],grid_shape[1],1,1 ]
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))  # shape = [ grid_shape[0],grid_shape[1],1,2 ]  = [13, 13, 1, 2] 这样子栅格的坐标信息已经获取到了

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])   # [ -1, 13, 13, 3, cls+5]

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))       # feats的最后一维度 [ x0,y0,w,h,confidence ]----->  中心坐标在每一个栅格的位置 -----> 归一化(0~1)的坐标位置
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats)) # w,h ------>
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh    # grid=[13,13,1,2] (只是形状的概念，里面的值就是坐标信息 即第i行第j列的栅格)，feats=[-1,13,13,3,cls+5](只是改变了网络输出的形状), box_xy=[-1,13,13,3,2](每一个栅格上面的中心坐标，归一化栅格大小 0~1)
    return box_xy, box_wh, box_confidence, box_class_probs  # box_wh=[-1,13,13,3,2](每一个栅格上面的宽和高， 归一化到原图像（416*416）


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""

    num_layers = len(yolo_outputs)

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32

    # print("yolo_outputs",yolo_outputs)
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5) # m:表示批次， T: box的数量， 5： 参数数量（x,y,w,h,cls）
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'  # 只返回 True 或 False
    num_layers = len(anchors)//3 # default setting  一共拥有 3个 feature map 做比较
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2      # box 中心点 坐标
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]             # box 的宽高
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]                  # box的中心点坐标 归一化 0-1
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]                  # box的宽高 归一化 0-1

    m = true_boxes.shape[0]    # batch_size
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]  # 配合 feature map
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes), # 5 ==> (x,y,w,h,confidence)
        dtype='float32') for l in range(num_layers)]   # zhe

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0) #   # anchor 的shape 由  （9，2）===> （1，9，2）
    anchor_maxes = anchors / 2.              #  以 中点 为圆心 [  [[x0,y0],[x1,y1],...,[x8,y8]]  ]
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0     # 主要是前面每一张 图片都补充满 T = 20个的box数量(当然是用0充当宽高的)

    for b in range(m):
        # Discard zero rows.   # boxes_wh.shape = (batch_size,20,2) ===>  wh.shape = (真实合理的box数量,2)
        wh = boxes_wh[b, valid_mask[b]]  # 通过布尔类型对数据进行筛选 # [ [x0,y0],[x1,y1],[x2,y2],...[xn,yn] ]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)           #                        [  [[x0,y0]], [[x1,y1]], [[x2,y2]]  ]
        box_maxes = wh / 2.  # 以中心点为原点  # [  [[x0,y0]], [[x1,y1]], [[x2,y2]]  ]  == > shape=(num_boxes,1,2)
        box_mins = -box_maxes
        # 该批 图片中gt 与 所有的anchor做IOU计算
        intersect_mins = np.maximum(box_mins, anchor_mins) # box_mins与anchor_mins 的shape分别为 (4,1,2),(1,9,2) ==> (4,9,2)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)           # shape=(4,9,2)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.) # shape=(4,9,2)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]                              # shape=(4,1)
        anchor_area = anchors[..., 0] * anchors[..., 1]                 # shape=(1,9)
        iou = intersect_area / (box_area + anchor_area - intersect_area)  # shape = (4,9) 4表示4个GT
                                                                           #               9表示9个anchor box  这意为着每个GT都分别于每一个anchor求一下IOU值
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)   # 返回的是   [3,3，3，0]   # 相当于为每一个GT　找到了最匹配它的　anchor_box的index
                                                                     # 3为选中的
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]: # anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32') # true_boxes‘shape = [batch_size,box数量,（x,y,w,h,cls）这5个参数]
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)   # k = 0  表示哪一个box（每一个feature map中的每一位置有3个待选的box）
                    c = true_boxes[b,t, 4].astype('int32')  # 类别
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]  # b表示 第b个图像，j表示x轴的栅格坐标，i表示y轴的栅格坐标，k表示3个box的哪一个，坐标变成归一化后的做白哦
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true

def my_box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half   # shape = (?,?,3,1,2)
    b1_maxes = b1_xy + b1_wh_half  # shape = (?,?,3,1,2)

    # Expand dim to apply broadcasting.
    b2_xy = b2[..., :2]       # shape = (1,?,2)
    b2_wh = b2[..., 2:4]      # shape = (1,?,2)
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins) # b1_mins.shape = (?,?,3,1,2)  ------  b2_mins.shape = (1,?,2)  =======> shape = (?,?,3,?,2)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou
def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half   # shape = (?,?,3,1,2)
    b1_maxes = b1_xy + b1_wh_half  # shape = (?,?,3,1,2)

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0) # shape = (1,?,4)
    b2_xy = b2[..., :2]       # shape = (1,?,2)
    b2_wh = b2[..., 2:4]      # shape = (1,?,2)
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins) # b1_mins.shape = (?,?,3,1,2)  ------  b2_mins.shape = (1,?,2)  =======> shape = (?,?,3,?,2)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

########################################
def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:
    举例时假设pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = K.pow(enclose_wh[..., 0], 2) + K.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方
    p2 = K.pow(boxes1[..., 0] - boxes2[..., 0], 2) + K.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。加上除0保护防止nan。
    atan1 = tf.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-9))
    atan2 = tf.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-9))
    v = 4.0 * K.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou
##########################################################################

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False, use_ciou_loss=True):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    use_ciou_loss: whether (or not) to use ciou as loss
    Returns
    -------
    loss: tensor, shape=(1,)

    '''

    num_layers = len(anchors)//3       # default setting
    yolo_outputs = args[:num_layers]   # 网络的输出 [  [batch_size,13,13,3*(5+num_cls)],[batch_size,26,26,3*(5+num_cls)],[batch_size,52,52,3*(5+num_cls)]   ]
    y_true = args[num_layers:]         # 标签值 [ [batch_size,13,13,3,5+num_cls],[batch_size,26,26,3,5+num_cls],[batch_size,52,52,3,5+num_cls]]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]     # 大的anchor 对应小的feature map
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))            # 反推出  输入网络时 图像的大小 例如: [416,416]
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]    #  [ [13,13],[26,26],[52,52] ]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]          # batch_size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    #########################################
    # with tf.Session() as sess1:
    #     m_int = m.eval()
    # sess1.close()

    #########################################
    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]         # 置信度  非0即1
        true_class_probs = y_true[l][..., 5:]     # 类别

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],  # 小的featrue <===搭配===> 大的anchor
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh]) # (?,13,13,3,4)

        # Darknet raw box to calculate loss.

        # GT 中心点(x,y)的偏移量  和神经网络预测值相对应（sigmoid(feats[...,:2])））
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid

        # GT对应的 宽和高   神经网络预测值对应的 feats[...,2:4]
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf

        # 防止 大框占的权重比例大， 小框占的权重比例小
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]


        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool') # shape = [?,13,13,3,1]
        def loop_body(b, ignore_mask):
            # 该行代码本身就有点多余，没有物体的GT box其值本来就是 0
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])   # 不含物体的神经网格区域给 剔除掉
            #                            shape =(13,13,3,4)      (13,13,3)   ====>   b 取消掉了batch_size 这个维度，l确定了 在13*13这个维度上的比较
            # 某一个图片上面的 pre box 与  某一个图片上的true_box, 的iou比值，
            iou = box_iou(pred_box[b], true_box) # pred_box[b].shape = (13,13,3,4),  true_box.shape = (box数量,4)  ===> iou.shape = (13,13,3,box数量)
            best_iou = K.max(iou, axis=-1)       # shape = (13,13,3)  ==== > 先获取 每个grid的每个尺度(大中小3个尺度，所以最后一个尺度)的最大IOU
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box))) # 把预测过程中，与真实box， IoU太低的预测框设置为忽略
            return b+1, ignore_mask
        # def loop_body(b, ignore_mask):
        #     ''' 经过个人修改的代码，可以训练，也可以收敛，其权重的效果未经过测试 '''
        #     # 该行代码本身就有点多余，没有物体的GT box其值本来就是 0
        #     true_box = y_true[l][b,...,0:4]
        #     iou = my_box_iou(pred_box[b], true_box)
        #     iou = K.expand_dims(iou, -1)
        #     best_iou = K.max(iou, axis=-1)
        #     ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box))) # 把预测过程中，与真实box， IoU太低的预测框设置为忽略
        #     return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()            # (?,?,?,3)  =====> (batch_size,13,13,3)
        ignore_mask = K.expand_dims(ignore_mask, -1) # (?,?,?,3,1)  =====> (batch_size,13,13,3,1)

        ##################按照自己的思路和习惯实现上述代码#######################

        # object_mask_bool = K.cast(object_mask, 'bool') # shape = [?,13,13,3,1]
        # want_ignore_mask_shape =  object_mask_bool.shape
        #
        # # ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=want_ignore_mask_shape)
        # ignore_mask = tf.ones(shape = [1,int(want_ignore_mask_shape[1]),int(want_ignore_mask_shape[2]),int(want_ignore_mask_shape[3])])
        # for b in range(int(1)):
        #     true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])   # 不含物体的神经网格区域给 剔除掉
        #     iou = box_iou(pred_box[b], true_box)
        #     best_iou = K.max(iou, axis=-1)
        #     ignore_mask = K.switch(best_iou < ignore_thresh,K.zeros_like(ignore_mask),ignore_mask)
        # ignore_mask = K.expand_dims(ignore_mask, -1)
        # print('ignore_mask_shape:',ignore_mask.shape)
        ########################################################################
        # K.binary_crossentropy is helpful to avoid exp overflow.   K.sigmoid(


        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask # 含有物体的 置信度 + 不含物体的 置信度
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
        #################################################
        if use_ciou_loss:  # 使用CIOU损失
            raw_true_box = y_true[l][...,0:4]
            print('pred_box:',pred_box)
            print('raw_true_box:',raw_true_box)
            ciou = K.expand_dims(bbox_ciou(pred_box, raw_true_box),axis = -1)
            ciou_loss = object_mask * box_loss_scale * (1 - ciou)
            ciou_loss = K.sum(ciou_loss) / mf  
            loss +=ciou_loss
        else:             # 使用IOU损失
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)  # 原始的
            # xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, K.sigmoid(raw_pred[..., 0:2]), from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
            xy_loss = K.sum(xy_loss) / mf
            wh_loss = K.sum(wh_loss) / mf
            loss += xy_loss + wh_loss
        # 通用的 置信度损失 和 类别损失
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += confidence_loss + class_loss   # 增加了add_ciou_loss

        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
