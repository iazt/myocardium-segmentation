import keras as K
from keras import Model
from keras.layers import Layer
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.applications.vgg16 import VGG16
from keras.layers import concatenate
from keras import Input
import numpy as np
from keras.layers import Conv2DTranspose
from keras.layers import Conv2D
from keras.backend import flatten
import tensorflow as tf
from keras_unet.losses import jaccard_distance
from keras_unet.metrics import iou, iou_thresholded
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.backend.clip(y_pred, K.backend.epsilon(), 1 - K.backend.epsilon())
        # calc
        loss = y_true * K.backend.log(y_pred) * weights
        loss = -K.backend.sum(loss, -1)
        return loss

    return loss

def per_class_iou(y_true, y_pred, smooth=1.):
    n_classes = y_pred.shape[3]
    total = 0
    for class_number in range(n_classes):
        y_true_f = K.backend.flatten(y_true[:, :, :, class_number])
        y_pred_f = K.backend.flatten(y_pred[:, :, :, class_number])
        intersection = K.backend.sum(y_true_f * y_pred_f)
        total += (intersection + smooth) / (K.backend.sum(y_true_f) + K.backend.sum(y_pred_f) - intersection + smooth)
    return total/3

class Gray2VGGInput(Layer):
    """Custom conversion layer"""
    def build(self, x):
        self.image_mean = K.backend.variable(value=np.array([103.939, 116.779, 123.68]).reshape([1,1,1,3]).astype('float32'),
                                     dtype='float32',
                                     name='imageNet_mean' )
        self.built = True
        return
    def call(self, x):
        rgb_x = concatenate([x,x,x], axis=-1 )
        norm_x = rgb_x - self.image_mean
        return norm_x

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (3,)


def UNET1_VGG16(img_rows=256, img_cols=256, keep_prob=0.3,loss = jaccard_distance, optimizer = Adam()):
    '''
    UNET with pretrained layers from VGG16
    '''
    def upsampleLayer(in_layer, concat_layer, input_size):
        '''
        Upsampling (=Decoder) layer building block
        Parameters
        ----------
        in_layer: input layer
        concat_layer: layer with which to concatenate
        input_size: input size fot convolution
        '''
        upsample = Conv2DTranspose(input_size, (2, 2), strides=(2, 2), padding='same')(in_layer)
        upsample = concatenate([upsample, concat_layer])
        conv = Conv2D(input_size, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(upsample)
        conv = BatchNormalization()(conv)
        conv = Dropout(0.3)(conv)
        conv = Conv2D(input_size, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(conv)
        conv = BatchNormalization()(conv)
        return conv

    #--------
    #INPUT
    #--------
    #batch, height, width, channels
    inputs_1 = Input((img_rows, img_cols, 1))

    #-----------------------
    #INPUT CONVERTER & VGG16
    #-----------------------
    inputs_3 = Gray2VGGInput(name='gray_to_rgb')(inputs_1)  #shape=(img_rows, img_cols, 3)
    base_VGG16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs_3,classes=3)

    #--------
    #DECODER
    #--------
    c1 = base_VGG16.get_layer("block1_conv2").output #(None, 256, 256, 64)
    c2 = base_VGG16.get_layer("block2_conv2").output #(None, 432, 616, 128)
    c3 = base_VGG16.get_layer("block3_conv2").output #(None, 216, 308, 256)
    c4 = base_VGG16.get_layer("block4_conv2").output #(None, 108, 154, 512)

    #--------
    #BOTTLENECK
    #--------
    c5 = base_VGG16.get_layer("block5_conv2").output #(None, 54, 77, 512)

    #--------
    #ENCODER
    #--------
    c6 = upsampleLayer(in_layer=c5, concat_layer=c4, input_size=512)
    c7 = upsampleLayer(in_layer=c6, concat_layer=c3, input_size=256)
    c8 = upsampleLayer(in_layer=c7, concat_layer=c2, input_size=128)
    c9 = upsampleLayer(in_layer=c8, concat_layer=c1, input_size=64)
    #--------
    #DENSE OUTPUT
    #--------
    out_channels=3
    outputs = Conv2D(out_channels, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[inputs_1], outputs=[outputs])

    #Freeze layers
    for layer in model.layers[:16]:
        layer.trainable = False

    print(model.summary())
    loss= weighted_categorical_crossentropy([0.1,100,1])
    model.compile(optimizer=optimizer,
                  loss=jaccard_distance,
                  metrics=[per_class_iou])

    return model