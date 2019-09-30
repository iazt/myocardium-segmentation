import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model
from keras_unet.metrics import iou, dice_coef, iou_thresholded
from keras_unet.models import custom_unet
from keras_unet.utils import get_augmented
from keras_unet.utils import plot_segm_history, plot_imgs


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    # If this doesn't work then install graphviz using its webpage
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

    # ---------------------------------------------------
    LOG_DIR = 'logs'
    directory = r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\data'
    model_filename = r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\weights\segm_model_using_focal_loss.h5'
    keep_training = True                            # True to train the model
    evaluate = True                                 # Evaluates the model in testing dataset and plot some results
    visualize_thresholded = True                    # True to visualize the testing results but applying a threshold
    threshold = 0.5
    plot_model_ = False                             # True to plot the model in a fixed image
    batch_size = 32
    steps_per_epoch = 10
    epochs = 100
    # --------------------------------------------------

    model = custom_unet((256, 256, 1),
                        num_classes=3,
                        use_batch_norm=True,
                        filters=8,
                        output_activation='softmax')
    model.summary()
    model.compile('adam', loss=binary_focal_loss(2., .25), metrics=[iou, iou_thresholded, dice_coef])
    if plot_model_:
        plot_model(model, show_shapes=True, rankdir='TB', to_file='model_vertical.png')

    # Callbacks definitions
    callback_checkpoint = ModelCheckpoint(
        model_filename,
        verbose=1,
        monitor='val_loss',
        save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)
    reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

    # Loading the weights
    try:
        model.load_weights(model_filename)
    except OSError:
        print("Weights not found!")

    if keep_training:
        # Original data to train
        print("Loading data")
        x_train = np.load(directory + r'\train_imgs.npy').astype('float32')
        x_train.resize((x_train.shape[0], 256, 256, 1))
        print("Training images loaded!")

        y_train = np.load(directory + r'\train_annot_v5.npy').astype('float32')
        print("Training annotations loaded!")

        # Stacked data for validation
        x_val = np.load(directory + r'\val_imgs.npy').astype('float32')
        x_val.resize((x_val.shape[0], 256, 256, 1))
        print("Validation images loaded!")

        y_val = np.load(directory + r'\val_annot_v5.npy').astype('float32')
        print("Validation annotations loaded!")

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

        print("Training...")
        # Training the model
        result = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                     validation_data=(x_val, y_val),
                                     callbacks=[callback_checkpoint, tensorboard_callback, early_stopping])
        plot_segm_history(result)

    if evaluate:
        # Load the test data
        y_test = np.load(directory + r'\test_annot_v5.npy').astype('float32')
        x_test = np.load(directory + r'\test_imgs.npy').astype('float32')
        x_test.resize((x_test.shape[0], 256, 256, 1))
        model.evaluate(x_test, y_test, verbose=1)

        # Makes some predictions
        y_pred = model.predict(x_test)
        plot_imgs(org_imgs=x_test, mask_imgs=y_test[:, :, :, 1], pred_imgs=y_pred[:, :, :, 1], nm_img_to_plot=5)

        if visualize_thresholded:
            y_pred[y_pred > threshold] = 1
            y_pred[y_pred < threshold] = 0
            plot_imgs(org_imgs=x_test, mask_imgs=y_test[:, :, :, 1], pred_imgs=y_pred[:, :, :, 1], nm_img_to_plot=5)

        # Getting the validation, training and testing final stats
        y1 = tf.placeholder(dtype='float32', shape=y_test.shape, name='y_true')
        y2 = tf.placeholder(dtype='float32', shape=y_test.shape, name='y_pred')
        op_iou = iou(y1, y2)
        op_iou_thresholded = iou_thresholded(y1, y2, threshold=threshold)
        op_dice_score = dice_coef(y1, y2)

        with tf.Session() as sess:
            iou_test = sess.run(op_iou, feed_dict={y1: y_test, y2: y_pred})
            iou_thresholded_test = sess.run(op_iou_thresholded, feed_dict={y1: y_test, y2: y_pred})
            dice_test = sess.run(op_dice_score, feed_dict={y1: y_test, y2: y_pred})

            print("----------- STATS -------------")
            print("IoU testing: {:.3f}".format(iou_test))
            print("IoU thresholded testing: {:.3f}".format(iou_thresholded_test))
            print("Dice score testing: {:.3f}".format(dice_test))


