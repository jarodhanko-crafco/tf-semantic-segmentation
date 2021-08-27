
import tensorflow as tf

from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras import backend as K

import tensorflow as tf

## IMPORTED from ##
"""
The implementation of VGG16/VGG19 based on Tensorflow.
Some codes are based on official tensorflow source codes.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""


def vgg(inputs, version='VGG19', output_stages='c5'):

    params = {'VGG16': [2, 2, 3, 3, 3],
              'VGG19': [2, 2, 4, 4, 4]}

    assert version in params
    version_params = params[version]
    dilation = [1, 1]

    _, h, w, _ = K.int_shape(inputs)

    # Block 1
    for i in range(version_params[0]):
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv' + str(i + 1))(inputs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    c1 = x

    # Block 2
    for i in range(version_params[1]):
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv' + str(i + 1))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    c2 = x

    # Block 3
    for i in range(version_params[2]):
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv' + str(i + 1))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    c3 = x

    # Block 4
    for i in range(version_params[3]):
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv' + str(i + 1),
                          dilation_rate=dilation[0])(x)
    if dilation[0] == 1:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    c4 = x

    # Block 5
    for i in range(version_params[4]):
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv' + str(i + 1),
                          dilation_rate=dilation[1])(x)
    if dilation[1] == 1:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    c5 = x

    outputs = {'c1': c1,
               'c2': c2,
               'c3': c3,
               'c4': c4,
               'c5': c5}

    if type(output_stages) is not list:
        return outputs[output_stages]
    else:
        return [outputs[ci] for ci in output_stages]


def conv_bn_relu(x, filters, kernel_size=1, strides=1):
    x = layers.Conv2D(filters, kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def asunet(input_shape=(256, 256, 1), num_classes=2, inputs=None):

    assert inputs is not None or input_shape is not None

    if inputs is None:
        assert isinstance(input_shape, tuple)
        inputs = layers.Input(shape=input_shape)  # 3

    c1, c2, c3, c4, c5 = vgg(inputs, 'VGG19', output_stages=['c1', 'c2', 'c3', 'c4', 'c5'])

    x = layers.Dropout(0.5)(c5)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = conv_bn_relu(x, 512, 2, strides=1)
    x = layers.Concatenate()([x, c4])
    x = conv_bn_relu(x, 512, 3, strides=1)
    x = conv_bn_relu(x, 512, 3, strides=1)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = conv_bn_relu(x, 256, 2, strides=1)
    x = layers.Concatenate()([x, c3])
    x = conv_bn_relu(x, 256, 3, strides=1)
    x = conv_bn_relu(x, 256, 3, strides=1)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = conv_bn_relu(x, 128, 2, strides=1)
    x = layers.Concatenate()([x, c2])
    x = conv_bn_relu(x, 128, 3, strides=1)
    x = conv_bn_relu(x, 128, 3, strides=1)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = conv_bn_relu(x, 64, 2, strides=1)
    x = layers.Concatenate()([x, c1])
    x = conv_bn_relu(x, 64, 3, strides=1)
    x = conv_bn_relu(x, 64, 3, strides=1)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(num_classes, 1, strides=1,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    outputs = x
    return Model(inputs, outputs)
