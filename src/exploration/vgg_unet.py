from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from src.exploration.vgg16 import get_vgg_encoder


class VggUnet(tf.keras.Model):

    def __init__(self, input_height, input_width, num_classes=10):
        super(VggUnet, self).__init__(name='my_model')
        self.num_classes = num_classes

        # Getting the VGG Model
        self._build(input_height, input_width)

    def _build(self, input_height, input_width):

        l1_skip_conn = True
        MERGE_AXIS = -1
        # Builds the U-Net since the VGG architecture
        model, levels = get_vgg_encoder(input_height=input_height, input_width=input_width)
        [f1, f2, f3, f4, f5] = levels

        # First u-net block
        model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
        model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='valid'))
        model.add(tf.keras.layers.BatchNormalization())

        # Second u-net block
        to_concatenate = tf.keras.layers.UpSampling2D((2, 2))
        model.add(to_concatenate)
        model.add(tf.keras.layers.concatenate(input=[to_concatenate, f3]))
        model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='valid'))
        model.add(tf.keras.layers.BatchNormalization())

        # Third u-net layer
        to_concatenate = tf.keras.layers.UpSampling2D((2, 2))
        model.add(to_concatenate)
        model.add(tf.keras.layers.concatenate([to_concatenate, f2], axis=MERGE_AXIS))
        model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='valid'))
        model.add(tf.keras.layers.BatchNormalization())
        to_concatenate = tf.keras.layers.UpSampling2D((2, 2))
        model.add(to_concatenate)

        if l1_skip_conn:
            model.add(tf.keras.layers.concatenate([to_concatenate, f1], axis=MERGE_AXIS))

        model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='valid'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(self.num_classes, (3, 3), padding='same'))

        self.model = model

model = VggUnet(224, 224)

