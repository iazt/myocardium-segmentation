import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from keras_unet.losses import jaccard_distance
from keras_unet.metrics import iou, dice_coef, iou_thresholded
from keras_unet.models import custom_unet
from keras_unet.utils import get_augmented
from keras_unet.utils import plot_segm_history, plot_imgs

import numpy as np

# If this doesn't work then install graphviz using its webpage
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# ---------------------------------------------------
LOG_DIR = 'logs'
directory = r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources'
model_filename = 'segm_model_v1.h5'
keep_training = False
evaluate = False
plot_model_ = True
batch_size = 32
steps_per_epoch = 10
epochs = 15
# --------------------------------------------------


model = custom_unet((256, 256, 1),
                    num_classes=3,
                    use_batch_norm=True,
                    filters=8,
                    output_activation='softmax')
model.summary()
model.compile('adam', loss=jaccard_distance, metrics=[iou, iou_thresholded, dice_coef])
if plot_model_:
    plot_model(model, rankdir='TB', to_file='model_vertical.png')

# Callbacks definitions
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True)
tensorboard_callback = TensorBoard(log_dir=LOG_DIR)

csv_logger = CSVLogger(directory+'training.log')

# Loading the weights
model.load_weights(model_filename)

if keep_training:
    # Original data to train
    x_train = np.load(directory + r'\train_imgs.npy').astype('float32')
    x_train.resize((x_train.shape[0], 256, 256, 1))

    y_train = np.load(directory + r'\train_annot_v5.npy').astype('float32')

    # Stacked data for validation
    x_val = np.load(directory + r'\val_imgs.npy').astype('float32')
    x_val.resize((x_val.shape[0], 256, 256, 1))

    y_val = np.load(directory + r'\val_annot_v5.npy').astype('float32')

    # Defining the data generator
    train_gen = get_augmented(
        x_train, y_train, batch_size=batch_size,
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
    # Training the model
    result = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=(x_val, y_val),
                                 callbacks=[callback_checkpoint, tensorboard_callback,csv_logger],
                                 use_multiprocessing=True)
    plot_segm_history(result)

if evaluate:
    # Load the test data
    y_test = np.load(directory + r'\test_annot_v5.npy').astype('float32')
    x_test = np.load(directory + r'\test_imgs.npy').astype('float32')
    x_test.resize((x_test.shape[0], 256, 256, 1))
    model.evaluate(x_test, y_test, verbose=1, use_multiprocessing=True)

    # Makes some predictions
    y_pred = model.predict(x_test)
    plot_imgs(org_imgs=x_test, mask_imgs=y_test[:, :, :, 1], pred_imgs=y_pred[:, :, :, 1], nm_img_to_plot=5)

