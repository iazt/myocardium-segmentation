import numpy as np
import matplotlib.pyplot as plt

"""
Little script for first data visualization of ACDC and York Database
"""


img_data = np.load('/Users/rudy/Documents/myocardium/resources/val_imgs.npy')
ann_data = np.load('/Users/rudy/Documents/myocardium/resources/val_annot.npy')

# Data description
total_imgs = img_data.shape[0]
width = img_data.shape[1]
height = img_data.shape[2]

print("N images: {}\nWidth: {}\nHeight: {}".format(total_imgs, width, height))

# Simple visualization
fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex='col', sharey='row')

idxs = np.random.randint(0, total_imgs, size=(2, 2))

for i in range(2):
    for j in range(2):
        axs[i][j].set(title="Img {} {}x{}".format(idxs[i][j], width, height))
        axs[i][j].imshow(img_data[idxs[i][j], :, :])
plt.show()

mask_visualization = True

if mask_visualization:

    # Plotting the mask
    n_img = idxs[0][0]
    selected_img = img_data[n_img, :, :].copy()

    r, g, b = np.zeros(shape=(width, height)), np.zeros(shape=(width, height)), np.zeros(shape=(width, height))

    # Makes a preview of the annotation
    r[ann_data[n_img, :, :] == 0] = 0
    g[ann_data[n_img, :, :] == 1] = 1.0
    b[ann_data[n_img, :, :] == 2] = 0.5
    annotation_preview = np.stack([r, g, b], axis=2)

    # Draw the segmentation into the image
    img_without_background = selected_img.copy()
    selected_img[ann_data[n_img, :, :] == 1] = 255
    selected_img[ann_data[n_img, :, :] == 2] = 50
    img_without_background[ann_data[n_img, :, :] == 0] = 0

    fig2, axs = plt.subplots(2, 2, figsize=(12, 12), sharex='col', sharey='row')
    axs[0][0].set(title='Original image: {}'.format(n_img))
    axs[0][0].imshow(img_data[n_img, :, :])

    axs[0][1].set(title='Segmentation mask and image')
    axs[0][1].imshow(selected_img)

    axs[1][0].set(title='Image without background')
    axs[1][0].imshow(img_without_background)

    axs[1][1].set(title='Segmentation mask')
    axs[1][1].imshow(annotation_preview)

    plt.show()

