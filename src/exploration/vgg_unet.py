from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from src.exploration.vgg16 import get_vgg_encoder


class VggUnet(tf.keras.Model):

    def __init__(self, input_height, input_width, num_classes=10):
        super(VggUnet, self).__init__(name='my_model')
        self.num_classes = num_classes

        # Getting the VGG Model
        self._build(input_height, input_width)

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

    def _build(self, input_height, input_width):

        IMAGE_ORDERING = 'channels_last'
        MERGE_AXIS = -1
        l1_skip_conn = False

        img_input, levels = get_vgg_encoder(input_height=input_height, input_width=input_width)
        [f1, f2, f3, f4, f5] = levels

        o = f4
        o = (tf.keras.layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.BatchNormalization())(o)

        o = (tf.keras.layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.concatenate([o, f3], axis=MERGE_AXIS))
        o = (tf.keras.layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.BatchNormalization())(o)

        o = (tf.keras.layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.concatenate([o, f2], axis=MERGE_AXIS))
        o = (tf.keras.layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.BatchNormalization())(o)

        o = (tf.keras.layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

        if l1_skip_conn:
            o = (tf.keras.layers.concatenate([o, f1], axis=MERGE_AXIS))

        o = (tf.keras.layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.BatchNormalization())(o)

        o = tf.keras.layers.Conv2D(self.num_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
        self.model = tf.keras.Model(img_input, o)

model = VggUnet(224, 224).model

