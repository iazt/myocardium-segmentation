from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from src.exploration.vgg16 import get_vgg_encoder
from keras_unet.metrics import iou_thresholded, iou, dice_coef
from keras.losses import categorical_crossentropy
from keras_unet.losses import jaccard_distance
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import layers
import keras
from keras_unet.utils import plot_imgs
from keras.backend.tensorflow_backend import set_session
from keras_unet.utils import get_augmented

# -------------------------------------------
keep_training = True
evaluate = True
directory = r'C:\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\data'
weights_directory = r'C:\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\weights'
model_name = 'vgg-unet-i256_3-o256_3_v0'
early_stopping = 3
patience_lr = 1
epochs = 100
batch_size = 8
dropout = 0.3
# ------------------------------------------

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    IMAGE_ORDERING = 'channels_last'
    MERGE_AXIS = -1
    l1_skip_conn = True

    # We generate our vgg model
    vgg_model, img_input, levels = get_vgg_encoder(input_height=256, input_width=256, input_depth=3)
    [f1, f2, f3, f4, f5] = levels

    # Here we connect the fourth convolutional block of vgg to the u-net
    o = f4
    o = layers.Conv2D(512, (3, 3),  activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = layers.BatchNormalization()(o)
    o = layers.Dropout(dropout)(o)

    # Expansion block
    o = layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = layers.concatenate([o, f3], axis=MERGE_AXIS)
    o = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = layers.BatchNormalization()(o)
    o = layers.Dropout(dropout)(o)

    # Expansion block
    o = layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = layers.concatenate([o, f2], axis=MERGE_AXIS)
    o = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = layers.BatchNormalization()(o)
    o = layers.Dropout(dropout)(o)

    # Expansion block
    o = layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    if l1_skip_conn:
        o = layers.concatenate([o, f1], axis=MERGE_AXIS)
    o = layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = layers.BatchNormalization()(o)
    o = layers.Dropout(dropout)(o)
    o = layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)

    # Expansion block
    o = layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = layers.BatchNormalization()(o)
    o = layers.Dropout(dropout)(o)
    o = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = layers.BatchNormalization()(o)
    o = layers.Conv2D(3, (1, 1), activation='softmax', padding='same', data_format=IMAGE_ORDERING)(o)

    # Creates the final block model
    vgg_unet_model = keras.Model(img_input, o, name="VGG-UNET")
    vgg_unet_model.summary()
    vgg_unet_model.compile(optimizer='adam', loss=jaccard_distance, metrics=[dice_coef])
    model = vgg_unet_model

    try:
        model.load_weights(weights_directory + '\\' + model_name + '.h5')
    except OSError:
        print("Weights not found!")
    except ValueError:
        print("Something had happened. The weights doesn't match the architecture.")

    if keep_training:

        # Load the data to train
        print("Loading data from {} ...".format(directory))
        img_train_data = np.load(directory + r'\train_imgs_v256_3npy.npy').astype('float32')
        annot_train_data = np.load(directory + r'\train_annot_v5.npy').astype('float32')
        img_val_data = np.load(directory + r'\val_imgs_v256_3npy.npy').astype('float32')
        annot_val_data = np.load(directory + r'\val_annot_v5.npy').astype('float32')
        print("Data loaded!")

        # Create useful callbacks
        early_stopper = EarlyStopping(patience=early_stopping, verbose=1)
        reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=patience_lr, verbose=1)
        checkpointer = ModelCheckpoint(weights_directory + f'\\{model_name}.h5', verbose=1, save_best_only=True)

        # Defining the data generator
        train_gen = get_augmented(
            img_train_data, annot_train_data, batch_size=batch_size,
            data_gen_args=dict(
                rotation_range=15.,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=50,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='constant'
            ))

        # Train model
        results = model.fit_generator(train_gen,
                                      validation_data=([img_val_data, annot_val_data]),
                                      steps_per_epoch=100, epochs=epochs,
                                      callbacks=[early_stopper, checkpointer, reduce_learning_rate])

    if evaluate:

        # Load testing data
        x_test = np.load(directory + r'\test_imgs_v256_3npy.npy').astype('float32')
        y_test = np.load(directory + r'\test_annot_v5.npy').astype('float32')

        # Getting the predictions for the testing set
        y_pred = model.predict(x_test)
        plot_imgs(org_imgs=x_test[500:, :, :, 1], mask_imgs=y_test[500:, :, :, 1], pred_imgs=y_pred[500:, :, :, 1], nm_img_to_plot=5)





