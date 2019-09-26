# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 02:15:48 2019

@author: ignac
"""
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def preprocessing(X): 
    X_shape=X.shape
    X=X.reshape(X_shape[0],X_shape[1],X_shape[2],1)
    
    X=tf.image.resize_images(X,
                      size=[224,224])
    
    return tf.concat([X,X,X],axis=3)
