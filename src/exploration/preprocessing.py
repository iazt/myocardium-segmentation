# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 02:15:48 2019

@author: ignac
"""
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def preprocessing(X,y): 
    X_shape=X.shape
    X=X.reshape(X_shape[0],X_shape[1],X_shape[2],1)
    
    X=tf.image.resize_images(X,
                      size=[224,224])
    
    datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip= True)
    

    datagen.fit(X)
    return datagen
