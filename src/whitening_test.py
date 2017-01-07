# coding=utf-8

# O proposito desse arquivo é ajudar a testar o whitening
# nas imagens.

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_img(img):
    """
    Draw an image
    :param img: numpy array
    """
    plt.imshow(img)
    plt.show()


def draw_multiple_images(images, num_lines, num_columns):
    """
    Draw multiple images (usually k-means centroids) in a grid

    :param images: numpy array with multiple images (each image is a numpy array)
    :param num_lines: number of lines in grid
    :param num_columns: number of columns in grid
    """

    img_is_flatten = True if len(images[0][0].shape) == 1 else False

    fig, subs = plt.subplots(num_lines, num_columns)
    for i in xrange(num_lines):
        for j in xrange(num_columns):
            img = images[i][j]

            if img_is_flatten:

                width = np.sqrt(img.shape[0])
                height = np.sqrt(img.shape[0])
                subs[i][j].imshow(img.reshape(width, height), cmap='Greys_r')

            else:
                subs[i][j].imshow(img, cmap='Greys_r')

            subs[i][j].axis('off')

    plt.show()


def whitenning(img):
    # implementar whitening aqui
    return img


img_path = "../girl-original.png"
imgw_path = "../girl-whitened.png"

img = cv2.imread(img_path)
imgw = cv2.imread(imgw_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgw = cv2.cvtColor(imgw, cv2.COLOR_BGR2RGB)

original_image_whitened = whitenning(img)
draw_multiple_images(np.array([[img, img], [imgw, original_image_whitened]]), 2, 2)
