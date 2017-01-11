# coding=utf-8

# O proposito desse arquivo é ajudar a testar o whitening
# nas imagens.

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


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


def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))  # [M x M]
    return ZCAMatrix


def whitenning(img):
    # implementar whitening aqui
    """
    Apply whitenning ZCA in an image
    :param img: nparray that is the image with three channels
    :return: nparray that is the image with three channels and whitened
    """
    b, g, r = cv2.split(img)
    bw = pca_white(b)
    gw = pca_white(g)
    rw = pca_white(r)

    img = cv2.merge((bw, gw, rw))
    return img

def pca_white(x):

    # plt.figure()
    # plt.plot(b[0,:], b[1,:], 'o', mec='blue', mfc='none')
    # plt.title('Raw data')
    # plt.show()

    # STEP 1a: Implement PCA to obtain the rotation matrix, U, which is
    # the eigenbases sigma.

    sigma = x.dot(x.T) / x.shape[1]
    U, S, Vh = linalg.svd(sigma)

    # plt.figure()
    # plt.plot([0, U[0,0]], [0, U[1,0]])
    # plt.plot([0, U[0,1]], [0, U[1,1]])
    # plt.plot(b[0,:], b[1,:], 'o', mec='blue', mfc='none')
    # plt.show()

    # STEP 1b: Compute xRot, the projection on to the eigenbasis

    xRot = U.T.dot(x)

    # plt.figure()
    # plt.plot(xRot[0,:], xRot[1,:], 'o', mec='blue', mfc='none')
    # plt.title('xRot')
    # plt.show()

    # STEP 2: Reduce the number of dimensions from 2 to 1

    k = 1
    xRot = U[:,0:k].T.dot(x)
    xHat = U[:,0:k].dot(xRot)

    # plt.figure()
    # plt.plot(xHat[0,:], xHat[1,:], 'o', mec='blue', mfc='none')
    # plt.title('xHat')
    # plt.show()

    # STEP 3: PCA Whitening

    epsilon = 1e-5
    xPCAWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T).dot(x)
    xZCAWhite = U.dot(xPCAWhite)

    # plt.figure()
    # plt.plot(xPCAWhite[0,:], xPCAWhite[1,:], 'o', mec='blue', mfc='none')
    # plt.title('xPCAWhite')
    # plt.show()
    return  xZCAWhite

img_path = "../girl-original.png"
imgw_path = "../girl-whitened.png"

img = cv2.imread(img_path)
imgw = cv2.imread(imgw_path)

cv2.normalize(img, img, alpha=0.0, beta=1.0, dtype=3, norm_type=cv2.NORM_MINMAX)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgw = cv2.cvtColor(imgw, cv2.COLOR_BGR2RGB)


original_image_whitened = whitenning(img)
draw_multiple_images(np.array([[img, img], [imgw, original_image_whitened]]), 2, 2)