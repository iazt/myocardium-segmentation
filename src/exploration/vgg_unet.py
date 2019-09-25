from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from .vgg16 import get_vgg_encoder


class VggUnet(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(VggUnet, self).__init__(name='my_model')
        self.num_classes = num_classes

        # Getting the VGG Model
        self.model = get_vgg_encoder()

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
