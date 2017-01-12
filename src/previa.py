import cPickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from image_utils import apply_whitening_in_image

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


def start_timer(name=None):
    if name is None:
        start_timer.start = time.time()
    else:
        end = time.time()
        print("Benchmark {} took: {} seconds.".format(name, end - start_timer.start))


def read_images_from_cifar_10(data_files, test_files, normalize=True, whitenning=True):
    """
    Read images from cifar 10 folder and prepare the train set, valid set and test set

    :param file_path: path for cifar 10 cpickled file
    :param normalize:
    :param whitenning:
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
        img_whitened = apply_whitening_in_image(img_normalized, True)
        img[0] = flatten_img(img_whitened)

    for img_1 in valid_set:
        img_unflatted = unflatten(img_1[0])
        img_normalized = normalize_img(img_unflatted)
        img_whitened = apply_whitening_in_image(img_normalized, True)
        img_1[0] = flatten_img(img_whitened)

    for img_2 in test_set:
        img_unflatted = unflatten(img_2[0])
        img_normalized = normalize_img(img_unflatted)
        img_whitened = apply_whitening_in_image(img_normalized)
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


def normalize_img(img):
    """
    Normalize values inside an image matrix

    :param img: numpy array
    :return: img: numpy array
    """
    return cv2.normalize(img, img, alpha=0.0, beta=1.0, dtype=3, norm_type=cv2.NORM_MINMAX)


def draw_multiple_images(images, num_lines, num_columns):
    """
    Draw multiple images (usually k-means centroids) in a grid

    :param images: numpy array with multiple images (each image is a numpy array)
    :param num_lines: number of lines in grid
    :param num_columns: number of columns in grid
    """

    img_is_flatten = True if len(images[0].shape) == 1 else False

    fig, subs = plt.subplots(num_lines, num_columns)
    for i in xrange(num_lines):
        for j in xrange(num_columns):
            img = images[i * j]

            if img_is_flatten:

                width = np.sqrt(img.shape[0])
                height = np.sqrt(img.shape[0])
                subs[i][j].imshow(img.reshape(width, height))

            else:
                subs[i][j].imshow(img)

            subs[i][j].axis('off')

    plt.show()


def draw_img(img, title):
    """
    Draw an image in a resizable window with title

    :param img: numpy array
    :param title: the name of window
    """
    cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)


def unflatten(img_flatten):
    img = img_flatten.reshape(3, 1024)
    r, g, b = img[0], img[1], img[2]
    r = r.reshape(32, 32)
    g = g.reshape(32, 32)
    b = b.reshape(32, 32)
    img = cv2.merge((b, g, r))
    return img


def get_label_names(idx):
    """
    Return the name of label related to the index number
    :param idx: indice number that whant to know the label of class
    :return: label of class
    """
    fo = open('../cifar-10/batches.meta', 'rb')
    lista = cPickle.load(fo)
    return lista['label_names'][idx]


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


def run():
    # data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    # test_files = ['test_batch']
    # train_set, valid_set, test_set = read_images_from_cifar_10(data_files, test_files)
    # img_2 = test_set[0][0]
    # img_2 = unflatten(img_2)
    # label_2 = test_set[0][1]
    # draw_img(img_2, 'image of test set do tipo {}'.format(label_2))
    # img_1 = train_set[100][0]
    # img = unflatten(img_1)
    # label_1 = train_set[100][1]
    # draw_img(img, 'image of train set do tipo {}'.format(label_1))
    img_file = '../girl-original.png'
    img = cv2.imread(img_file)
    # img_matplot = plt.imread(img_file)
    img_whitened = apply_whitening_in_image(img, True)
    # img_whitened_mp = apply_whitening_in_image(img_matplot)
    draw_img(img, 'original')
    draw_img(img_whitened, 'whitened')
    # normalized = normalize_img(img_2)
    # draw_img(normalized)
    # img_nova = flatten_img(normalized)
    # print len(img_nova)
    # print img_nova
    # print type(img_nova)
    # oi = unflatten(img_nova)
    # draw_img(oi)


if __name__ == '__main__':
    run()
