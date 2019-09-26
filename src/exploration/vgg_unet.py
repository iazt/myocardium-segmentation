from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from src.exploration.vgg16 import get_vgg_encoder
from src.exploration.utils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class VggUnet(tf.keras.Model):

    def __init__(self, input_height, input_width, num_classes=10):
        super(VggUnet, self).__init__(name='Vgg16-unet')
        self.num_classes = num_classes

        # Getting the VGG Model
        self._load_data()
        self._build(input_height, input_width)

    def _load_data(self):

        directory = r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources'
        print("Loading data from {} ...".format(directory))
        self.img_train_data = np.load(directory + r'\train_imgs.npy').astype('float32')
        self.annot_train_data = np.load(directory + r'\train_annot.npy').astype('float32')
        self.img_val_data = np.load(directory + r'\val_imgs.npy').astype('float32')
        self.annot_val_data = np.load(directory + r'\val_annot.npy').astype('float32')
        self.img_test_data = np.load(directory + r'\test_imgs.npy').astype('float32')
        self.annot_test_data = np.load(directory + r'\test_annot.npy').astype('float32')
        print("Data loaded!")

    def _build(self, input_height, input_width):

        IMAGE_ORDERING = 'channels_last'
        MERGE_AXIS = -1
        l1_skip_conn = True

        img_input, levels = get_vgg_encoder(input_height=input_height, input_width=input_width)
        [f1, f2, f3, f4, f5] = levels

        # Here we connect the fourth convolutional block of vgg to the u-net
        o = f4
        o = (tf.keras.layers.Conv2D(512, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.BatchNormalization())(o)

        # Expansion block
        o = (tf.keras.layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.concatenate([o, f3], axis=MERGE_AXIS))
        o = (tf.keras.layers.Conv2D(256, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.BatchNormalization())(o)

        # Expansion block
        o = (tf.keras.layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.concatenate([o, f2], axis=MERGE_AXIS))
        o = (tf.keras.layers.Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.BatchNormalization())(o)

        # Expansion block
        o = (tf.keras.layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        if l1_skip_conn:
            o = (tf.keras.layers.concatenate([o, f1], axis=MERGE_AXIS))
        o = (tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.BatchNormalization())(o)
        o = tf.keras.layers.Conv2D(32, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)

        # Expansion block
        o = (tf.keras.layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.Conv2D(16, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.Conv2D(16, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
        o = (tf.keras.layers.Conv2D(3, (1, 1), activation='softmax', padding='same', data_format=IMAGE_ORDERING))(o)

        # Creates the final block model
        vgg_unet_model = tf.keras.Model(img_input, o)
        # output_shape = vgg_unet_model.output_shape
        # input_shape = vgg_unet_model.input_shape
        # n_classes = output_shape[3]
        # model = tf.keras.Model(img_input, o)
        # model.output_width = output_shape[2]
        # model.output_height = output_shape[1]
        # model.n_classes = n_classes
        # model.input_height = input_shape[1]
        # model.input_width = input_shape[2]
        # model.model_name = "VGG16-UNET"
        self.model = vgg_unet_model
        self.model.summary()
        self.model.compile(optimizer='adam', loss=jaccard_distance_loss, metrics=[dice_coef])

    def train(self, epochs, early_stopper, patience_lr, model_name):
        early_stopper = EarlyStopping(patience=early_stopper, verbose=1)
        reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=patience_lr, verbose=1)
        checkpointer = ModelCheckpoint(model_name + '.h5', verbose=1, save_best_only=True)
        checkpointer_train = ModelCheckpoint(model_name + 'best_train.h5', monitor='loss', verbose=1,
                                             save_best_only=True)
        results = self.model.fit(self.img_train_data, self.annot_train_data,
                                 validation_data=([self.img_val_data, self.annot_val_data]),
                                 batch_size=5, epochs=epochs,
                                 callbacks=[early_stopper, checkpointer, checkpointer_train, reduce_learning_rate])
        return results


unet = VggUnet(224, 224)
results = unet.train(10, 3, 3, '1Model')
