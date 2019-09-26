import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

"""
This script loads the input image and annotation data and save it on (224, 224, 3) shape
"""


def get_folder_and_name(path):

    separator = "\\"
    s = path.split(separator)
    directory = "\\".join(s[:-1])
    name = s[-1].split(".")[0]
    return directory, name


def get_image_tensor(paths):

    for path in paths:
        directory, name = get_folder_and_name(path)
        data = np.load(path)
        X = []
        for img_idx in range(data.shape[0]):
            img = data[img_idx]
            # Stacking the grayscale feature map and resizing the image to (224, 224, 3)
            with_channels = np.rollaxis(np.stack([img, img, img]), 0, 3)
            X.append(cv2.resize(with_channels, (224, 224)))
        X = np.array(X)
        np.save(directory + "\\" + name + "_v2.npy", X)


def get_ann_tensor(paths):

    width = 256
    height = 256
    for path in paths:
        directory, name = get_folder_and_name(path)
        data = np.load(path)
        X = []
        for ann_idx in range(data.shape[0]):
            ann = data[ann_idx]
            r, g, b = np.zeros(shape=(width, height)), np.zeros(shape=(width, height)), np.zeros(shape=(width, height))
            r[ann == 0] = 1
            g[ann == 1] = 1
            b[ann == 2] = 1
            annotation = np.stack([r, g, b], axis=2)
            X.append(cv2.resize(annotation, (224, 224)))
        X = np.array(X)
        np.save(directory + "\\" + name + "_v2.npy", X)


img_paths = [r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\train_imgs.npy',
             r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\val_imgs.npy',
             r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\test_imgs.npy']

ann_paths = [r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\train_annot.npy',
             r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\val_annot.npy',
             r'\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\test_annot.npy']

get_ann_tensor(ann_paths)
get_image_tensor(img_paths)