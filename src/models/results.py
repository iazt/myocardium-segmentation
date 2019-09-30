import matplotlib.pyplot as plt
import numpy as np
from keras_unet.utils import plot_imgs

directory = r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\data'
y_pred = np.load('preds.npy').astype('float32')
y_test = np.load(directory + r'\test_annot_v5.npy').astype('float32')
x_test = np.load(directory + r'\test_imgs.npy').astype('float32')
x_test.resize((x_test.shape[0], 256, 256, 1))

start = 304
end = 900
plot_imgs(org_imgs=x_test[start:end, :, :], mask_imgs=y_test[start:end, :, :, 1], pred_imgs=y_pred[start:end, :, :, 1],
          nm_img_to_plot=3)

