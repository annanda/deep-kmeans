import cv2
import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt


def read_images_from_mnist(file_path, normalize=True, whitenning=True):
    """read image from mnist folder

    :param file_path: path for mnist cpickled file
    :param normalize:
    :param whitenning:
    :return: train_set (50k examples): [ [list of images], [list of labels related with each image]  ]
    :return: valid_set (10k examples): [ [list of images], [list of labels related with each image]  ]
    :return: test_set (10k examples): [ [list of images], [list of labels related with each image]  ]

    note: image are normalized and whitenning following paper's
            instructions (http://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf)
    """

    f = gzip.open(file_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    for i in xrange(len(train_set[0])):
        img = np.array(train_set[0][i]).reshape(28,28)

        if normalize:
            img = normalize_img(img)

        if whitenning:
            img = whitenning_img(img)

        train_set[0][i] = img.reshape(784)

    for i in xrange(len(valid_set[0])):
        img = np.array(valid_set[0][i]).reshape(28,28)

        if normalize:
            img = normalize_img(img)

        if whitenning:
            img = whitenning_img(img)

        valid_set[0][i] = img.reshape(784)

    for i in xrange(len(test_set[0])):
        img = np.array(test_set[0][i]).reshape(28,28)

        if normalize:
            img = normalize_img(img)

        if whitenning:
            img = whitenning_img(img)

        test_set[0][i] = img.reshape(784)

    return train_set, valid_set, test_set


def normalize_img(img):
    """normalize values inside an image matrix

    :param img: numpy array
    :return: img: numpy array
    """
    return cv2.normalize(img, img, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)


def whitenning_img(img):
    """remove content (filter) image using a sobel filter

    :param img: numpy array
    :return: numpy array
    """
    return cv2.Sobel(img,cv2.CV_32F,1,1,ksize=5)


def sampling_image(img, window_size=(5,5)):
    """get a sample slice from a image

    :param img: numpy array
    :param window_size: a tuple (width, height) with the sample size
    :return: img: numpy array
    """
    x = np.random.randint(0, img.shape[0]-window_size[0])
    y = np.random.randint(0, img.shape[1]-window_size[1])
    return img[x : x + window_size[0], y: y + window_size[1]].flatten()


def generate_samples(data_set, num_samples, window_size=(5,5)):
    """from a set of images, generate many samples (slices)

    :param data_set: numpy array with multiple images (each image is a numpy array)
    :param num_samples: number of samples that must be generated
    :param window_size: tuple (width, height) with the sample size
    :return: img: numpy array with multiple image samples
    """
    samples = []
    for i in xrange(num_samples):
        random_img_posi = np.random.randint(0, len(data_set))
        img = data_set[random_img_posi]
        sample = sampling_image(img.reshape(28,28), window_size=window_size )
        samples.append(sample)
    return np.array(samples)


def draw_multiple_images(images, num_lines, num_columns):
    """draw multiple images (usually k-means centroids) in a grid

    :param images: numpy array with multiple images (each image is a numpy array)
    :param num_lines: number of lines in grid
    :param num_columns: number of columns in grid
    """

    img_is_flatten = True if len(images[0].shape) == 1 else False

    fig, subs = plt.subplots(num_lines, num_columns)
    for i in xrange(num_lines):
        for j in xrange(num_columns):
            img = images[i*j]

            if img_is_flatten:

                width = np.sqrt(img.shape[0])
                height = np.sqrt(img.shape[0])
                subs[i][j].imshow(img.reshape(width, height),  cmap='Greys_r')

            else:
                subs[i][j].imshow(img,  cmap='Greys_r')

            subs[i][j].axis('off')

    plt.show()


def draw_img(img):
    """ draw an image

    :param img: numpy array
    """
    plt.imshow(img, cmap='Greys_r')
    plt.show()


def convolve_image(img, filter):
    """apply a filter over a image

    :param img: numpy array representing original image
    :param filter: numpy array representing the filter that will be apply
    :return: img: numpy array with a filtered image
    """
    return cv2.filter2D(img, -1, filter)