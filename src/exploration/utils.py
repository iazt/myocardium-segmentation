from keras import backend as K


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    loss = (1 - jac) * smooth
    return loss


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


# img_data = np.load(r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\val_imgs.npy')
# ann_data = np.load(r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\val_annot.npy')
#
# # Data description
# total_imgs = img_data.shape[0]
# width = img_data.shape[1]
# height = img_data.shape[2]
#
# X = []
# for img_idx in range(img_data.shape[0]):
#     img = img_data[img_idx]
#     X.append(np.rollaxis(np.stack([img, img, img]), 0, 3))
# X = np.array(X)
#
# print("N images: {}\nWidth: {}\nHeight: {}".format(total_imgs, width, height))
# img = X[0, :, :, :].astype('float32')
# mask = X[0, :, :, :].astype('float32')
# plt.imshow(img[:, :, 2])
# plt.show()