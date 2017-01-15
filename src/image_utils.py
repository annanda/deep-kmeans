# coding=utf-8
import cPickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.cluster import MiniBatchKMeans

LABEL_AIRPLANE = 0
LABEL_AUTOMOBILE = 1
LABEL_BIRD = 2
LABEL_CAT = 3
LABEL_DEER = 4
LABEL_DOG = 5
LABEL_FROG = 6
LABEL_HORSE = 7
LABEL_SHIP = 8
LABEL_TRUCK = 9


def read_images_from_cifar_10(data_files, test_files):
    """
    Read images from cifar 10 folder and prepare the train set, valid set and test set

    :param data_files: path for cifar 10 cpickled data files
    :param test_files: path for cifar 10 cpickled test file
    :return: train_set (8k examples): numpy ndarray shape(8000, 2). row[i][0] = [image], row[i][1] = label
    :return: valid_set (2k examples): numpy ndarray shape(2000, 2). row[i][0] = [image], row[i][1] = label
    :return: test_set (2k examples): numpy ndarray shape(2000, 2). row[i][0] = [image], row[i][1] = label
    """

    cat_class = get_some_class(data_files, LABEL_CAT)
    train_set = cat_class[:4000]
    valid_set = cat_class[4000:]

    bird_class = get_some_class(data_files, LABEL_BIRD)
    train_set.extend(bird_class[:4000])
    valid_set.extend((bird_class[4000:]))

    cat_test = get_some_class(test_files, LABEL_CAT)
    test_set = cat_test

    bird_test = get_some_class(test_files, LABEL_BIRD)
    test_set.extend(bird_test)

    for img in train_set:
        img_unflatted = unflatten(img[0])
        img_normalized = normalize_img(img_unflatted)
        img_whitened = whiten_image(img_normalized, True)
        img[0] = flatten_img(img_whitened)

    for img_1 in valid_set:
        img_unflatted = unflatten(img_1[0])
        img_normalized = normalize_img(img_unflatted)
        img_whitened = whiten_image(img_normalized, True)
        img_1[0] = flatten_img(img_whitened)

    for img_2 in test_set:
        img_unflatted = unflatten(img_2[0])
        img_normalized = normalize_img(img_unflatted)
        img_whitened = whiten_image(img_normalized)
        img_2[0] = flatten_img(img_whitened)

    return train_set, valid_set, test_set


def get_some_class(list_files, class_name):
    final_array = []
    for file in list_files:

        dict = unpickle('../cifar-10/' + file)

        example_list = dic_to_array(dict)

        for example in example_list:
            if example[1] == class_name:
                final_array.append(example)

    return final_array


def unpickle(file):
    """
    Unpickle a pickled python object that contains 10000 images and its labels

    :param file: path to the file
    :return: dict with two elements:
        data: a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
        The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
        The image is stored in row-major order, so that the first 32 entries of the array are the red channel values
        of the first row of the image.
        label: a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the
        ith image in the array data.
    """
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def dic_to_array(dic):
    list_images = dic['data']
    list_labels = dic['labels']
    # make association between each image with its label
    array = zip(list_images, list_labels)
    array = np.array(list(array))

    return array


def normalize_img(img):
    """
    Normalize values inside an image matrix

    :param img: numpy array
    :return: img: numpy array
    """
    return cv2.normalize(img, img, alpha=0.0, beta=1.0, dtype=3, norm_type=cv2.NORM_MINMAX)


def unflatten(img_flatten):
    img = img_flatten.reshape(3, 1024)
    r, g, b = img[0], img[1], img[2]
    r = r.reshape(32, 32)
    g = g.reshape(32, 32)
    b = b.reshape(32, 32)
    img = cv2.merge((b, g, r))
    return img


def flatten_img(img):
    b, g, r = cv2.split(img)
    b = b.reshape(1024)
    g = g.reshape(1024)
    r = r.reshape(1024)
    b = list(b)
    g = list(g)
    r = list(r)
    img = r + g + b
    return np.array(img)


def draw_image(img, title="image"):
    """
    Draw an image in a resizable window with title

    :param img: numpy array
    :param title: the name of window
    """
    cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)

    # plt.imshow(img)
    # plt.show()


def whiten_image(img, is_zca=True):
    """
    Apply whitenning ZCA in an image
    :param img: nparray that is the image with three channels
    :param is_zca: bool value that indicates if whitening is ZCA. If value is False,
    whitening is PCA type.
    :return: nparray that is the image with three channels and whitened
    """
    b, g, r = cv2.split(img)
    b = whiten_channel(b, is_zca)
    g = whiten_channel(g, is_zca)
    r = whiten_channel(r, is_zca)

    img = cv2.merge((b, g, r))
    return img


def whiten_channel(channel, is_zca=True):
    """
    Funcao responsavel por receber uma imagem preto e branco (que também pode ser um canal)
    e retornar a matriz equivalente a mesma, porem esbranquicada.
    """

    width, height = channel.shape

    flat_channel = channel.reshape(1, channel.size)
    flat_channel = flat_channel.astype('float64')  # Evita problemas de overflow em algumas contas quando se usa int.

    if is_zca:
        whitened_flat_channel = zca_whitening(flat_channel)
    else:
        whitened_flat_channel, _ = pca_whitening(flat_channel)

    whitened_channel = whitened_flat_channel.reshape(width, height)

    return whitened_channel


def pca_whitening(x):
    """Funcao responsavel por receber a matriz no formato 1 x N*M de valores float e realizar os passos do
    algoritmo de branqueamento do PCA. Sao eles: calcular sigma e seus autovalores (matriz U de rotacao, etc),
    achar xRot (dados rotacionados), xHat(dados em dimensao reduzida 1) e, finalmente, computar a matriz PCA
    utilizando a formula estabelecida. Retorna a matriz PCA e U."""

    sigma = x.dot(x.T) / x.shape[1]
    U, S, Vh = linalg.svd(sigma)

    epsilon = 1e-5

    # noinspection PyTypeChecker
    x_pca_white = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T).dot(x)  # formula do PCAWhitening
    return x_pca_white, U


def zca_whitening(x):
    """Funcao responsavel por retornar a matriz ZCAWhitening a partir da PCAWhitening e U, atraves
    da formula estabelecida (UxPCAWhite)."""

    x_pca_white, U = pca_whitening(x)
    x_zca_white = U.dot(x_pca_white)  # formula da ZCAWhitening
    return x_zca_white


def get_random_patch_of_image(img, patch_width, patch_height):
    y = np.random.randint(0, img.shape[0] - patch_height)
    x = np.random.randint(0, img.shape[1] - patch_width)
    return img[y:y + patch_height, x:x + patch_width]


def get_random_patches_of_image(img, patch_width, patch_height, num_patches):
    patches = []
    for i in xrange(num_patches):
        patches.append(get_random_patch_of_image(img, patch_width, patch_height))

    return np.array(patches)


def get_random_patches_of_images(images, patch_width, patch_height, num_patches_per_image):
    patches = []
    for image in images:
        patches.extend(get_random_patches_of_image(image, patch_width, patch_height, num_patches_per_image))

    return np.array(patches)


def draw_images_with_matplot(image_matrix):
    num_lines = image_matrix.shape[0]
    num_columns = image_matrix.shape[1]

    fig, subs = plt.subplots(num_lines, num_columns)
    for i in xrange(num_lines):
        for j in xrange(num_columns):
            img = image_matrix[i][j]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if num_lines == 1:
                subs[j].imshow(img, cmap='Greys_r')
                subs[j].axis('off')
            else:
                subs[i][j].imshow(img, cmap='Greys_r')
                subs[i][j].axis('off')

    plt.show()


def draw_images_with_matplot_array_images(image_array):
    num_lines = image_array.shape[0] / 2
    num_columns = image_array.shape[1]

    fig, subs = plt.subplots(num_lines, num_lines)
    cont = 0
    for i in xrange(num_lines):
        for j in xrange(num_lines):
            img = image_array[cont]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            subs[i][j].imshow(img, cmap='Greys_r')
            subs[i][j].axis('off')
            cont += 1

    plt.show()


def generate_centroids_from_dataset(data_set, centroids_size, num_centroids):
    """generate a set of centroids from image

    :param data_set: a numpy array of images (numpy array)
    :param centroids_size: int that will determinate a tuple (width, height) with the sample size
    :param num_centroids: the number of centroids that will be generated
    :return: centroids: numpy array with the k-means generated centroids
    """

    images = []
    for flat_image in data_set.x:
        image = flat_image.reshape(32, 32, 3)
        images.append(image)

    # GENERATING PATCHES
    patches = get_random_patches_of_images(images, patch_width=centroids_size, patch_height=centroids_size,
                                           num_patches_per_image=2)

    data_to_fit = []
    for patch in patches:
        flat_patch = patch.flatten()
        data_to_fit.append(flat_patch)

    # FITTING K-MEANS
    kmeans_model = MiniBatchKMeans(n_clusters=num_centroids,
                                   batch_size=100,
                                   n_init=10,
                                   compute_labels=False,
                                   max_no_improvement=10,
                                   verbose=True)
    kmeans_model.fit(data_to_fit)

    centroids = kmeans_model.cluster_centers_

    return centroids
