import tensorflow as tf
import numpy as np

"""
This script transform a tensor of images from gray scale to RGB repeating the gray values in each RGB channel.
"""

output_postfix = '_v256_3'

input_paths = [
    r'C:\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\data\train_imgs.npy',
    r'C:\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\data\val_imgs.npy',
    r'C:\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\data\test_imgs.npy'
]


def get_folder_and_name(path):

    separator = "\\"
    s = path.split(separator)
    directory = "\\".join(s[:-1])
    name = s[-1].split(".")[0]
    return directory, name


for path in input_paths:
    data = np.load(path)
    data.resize((data.shape[0], 256, 256, 1))
    x = tf.placeholder(dtype='float32', name='data', shape=(data.shape[0], 256, 256, 1))
    op = tf.image.grayscale_to_rgb(x)
    folder, name = get_folder_and_name(path)
    with tf.Session() as sess:
        resized = sess.run(op, feed_dict={x: data})
        np.save(folder + "\\" + name + output_postfix + "npy", resized)



def plot_image_result(index, img_true, img_pred, title=''):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    axs[0].imshow(img_true, cmap='gray')
    axs[0].set(title=f'Ground truth {index}')
    axs[1].imshow(img_pred, cmap='gray')
    axs[1].set(title=f'Pred {index}')
    plt.savefig('exp.png')
    plt.show()


def np_iou(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)


def np_dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def np_iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = 0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)


def extract_statistics(y_test, y_pred, plot_examples=False, plot_best_worst=False):
    # Plotting 5 random samples, mask number k
    if plot_examples:
        selection = np.random.randint(0, y_test.shape[0], size=2)
        for i in selection:
            plot_image_result(i, y_test[i, :, :, 1], y_pred[i, :, :, 1])

    for metric in [np_iou, np_dice_coef, np_iou_thresholded]:

        print(f"USING {metric}")

        worst_result, worst_iou, best_iou, best_result = None, None, None, None
        # Extracts the metric per class, applies mean, and then applies mean over images scores
        metric_result = []
        for class_number in [0, 1, 2]:
            class_scores = []
            for i in range(y_test.shape[0]):
                class_scores.append(metric(y_test[i, :, :, class_number], y_pred[i, :, :, class_number]))

            class_mean_iou = np.mean(class_scores)
            class_std_iou = np.std(class_scores)
            print(f"For class {class_number}: {class_mean_iou}+-{class_std_iou}")
            metric_result.append(class_mean_iou)

            if class_number == 1:
                worst_result, worst_iou = np.argmin(class_scores), np.min(class_scores)
                best_result, best_iou = np.argmax(class_scores), np.max(class_scores)
        print("Mean and std of scores per class: {}".format(np.mean(metric_result)))

        # Extracts the metric for image and the applies mean
        images_scores = []
        for i in range(y_test.shape[0]):
            images_scores.append(metric(y_test[i, :, :, :], y_pred[i, :, :, :]))
        mean = np.mean(images_scores)
        std = np.std(images_scores)
        print(f"Global IoU: {mean} +/- {std}")

        if plot_best_worst:
            plot_image_result(worst_result,
                              y_test[worst_result, :, :, 1],
                              y_pred[worst_result, :, :, 1],
                              title=f'Worst result\nMyocardium iou: {worst_iou:.3f} - Image IoU: {images_scores[worst_result]:.3f}')
            plot_image_result(best_result,
                              y_test[best_result, :, :, 1],
                              y_pred[best_result, :, :, 1],
                              title=f'Best result\nMyocardium iou: {best_iou:.3f} - Image IoU: {images_scores[best_result]:.3f}')

        print("------------------------------")
