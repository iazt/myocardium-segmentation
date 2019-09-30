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


