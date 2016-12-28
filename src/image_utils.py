import cv2
import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt


def read_images_from_mnist(file_path, normalize=True, whitenning=True):
    """
    Read image from mnist folder

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
        img = np.array(train_set[0][i]).reshape(28, 28)

        if normalize:
            img = normalize_img(img)

        if whitenning:
            img = whitenning_img(img)

        train_set[0][i] = img.reshape(784)

    for i in xrange(len(valid_set[0])):
        img = np.array(valid_set[0][i]).reshape(28, 28)

        if normalize:
            img = normalize_img(img)

        if whitenning:
            img = whitenning_img(img)

        valid_set[0][i] = img.reshape(784)

    for i in xrange(len(test_set[0])):
        img = np.array(test_set[0][i]).reshape(28, 28)

        if normalize:
            img = normalize_img(img)

        if whitenning:
            img = whitenning_img(img)

        test_set[0][i] = img.reshape(784)

    return train_set, valid_set, test_set


def read_images_from_cifar_10(data_files, test_files, normalize=True, whitenning=True):
    """
    Read images from cifar 10 folder and prepare the train set, valid set and test set

    :param data_files: list of Strings that contains the names of cifar data files
    :param test_files: list of Strings that contains the names of cifar test files
    :param normalize:
    :param whitenning:
    :return: train_set (8k examples from two classes): numpy ndarray with shape (8000, 2). row[i][0] = [image], row[i][1] = label
    :return: valid_set (2k examples from two classes): numpy ndarray with shape (2000, 2). row[i][0] = [image], row[i][1] = label
    :return: test_set (2k examples from two classes): numpy ndarray with shape (2000, 2). row[i][0] = [image], row[i][1] = label
    """

    cat_class = get_some_class(data_files, 'cat')
    train_set = cat_class[:4000]
    valid_set = cat_class[4000:]

    bird_class = get_some_class(data_files, 'bird')
    train_set.extend(bird_class[:4000])
    valid_set.extend((bird_class[4000:]))

    cat_test = get_some_class(test_files, 'cat')
    test_set = cat_test

    bird_test = get_some_class(test_files, 'bird')
    test_set.extend(bird_test)

    return train_set, valid_set, test_set


def dic_to_array(dic):
    """
    Transform dictionary which keys are data and label into an array
    :param dic:dictionary which keys are data and label.
            data has the image and label has the class which that image belong to.
    :return: array: numpy ndarray which shape is (3072, 2) that contains
    """
    list_images = dic['data']
    list_labels = dic['labels']
    # make association between each image with its label
    array = zip(list_images, list_labels)
    array = np.array(list(array))

    return array


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


def normalize_img(img):
    """
    Normalize values inside an image matrix

    :param img: numpy array
    :return: img: numpy array
    """
    return cv2.normalize(img, img, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)


def whitenning_img(img):
    """
    Remove content (filter) image using a sobel filter

    :param img: numpy array
    :return: numpy array
    """
    return cv2.Sobel(img, cv2.CV_32F, 1, 1, ksize=5)


def sampling_image(img, window_size=(5, 5)):
    """
    Get a sample slice from a image

    :param img: numpy array
    :param window_size: a tuple (width, height) with the sample size
    :return: img: numpy array
    """
    x = np.random.randint(0, img.shape[0]-window_size[0])
    y = np.random.randint(0, img.shape[1]-window_size[1])
    return img[x: x + window_size[0], y: y + window_size[1]].flatten()


def generate_samples(data_set, num_samples, window_size=(5, 5)):
    """
    From a set of images, generate many samples (slices)

    :param data_set: numpy array with multiple images (each image is a numpy array)
    :param num_samples: number of samples that must be generated
    :param window_size: tuple (width, height) with the sample size
    :return: img: numpy array with multiple image samples
    """
    samples = []
    for i in xrange(num_samples):
        random_img_posi = np.random.randint(0, len(data_set))
        img = data_set[random_img_posi]
        sample = sampling_image(img.reshape(28, 28), window_size=window_size)
        samples.append(sample)
    return np.array(samples)


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
            img = images[i*j]

            if img_is_flatten:

                width = np.sqrt(img.shape[0])
                height = np.sqrt(img.shape[0])
                subs[i][j].imshow(img.reshape(width, height),  cmap='Greys_r')

            else:
                subs[i][j].imshow(img,  cmap='Greys_r')

            subs[i][j].axis('off')

    plt.show()


def get_label_names(idx):
    """
    Return the name of label related to the index number
    :param idx: indice number that whant to know the label of class
    :return: label of class
    """
    fo = open('../cifar-10/batches.meta', 'rb')
    lista = cPickle.load(fo)
    return lista['label_names'][idx]


def get_index_from_name_label(label):
    """
    Return the index of specific class name
    :param label: String name of the class
    :return: Inteiro index of class
    """
    fo = open('../cifar-10/batches.meta', 'rb')
    lista = cPickle.load(fo)
    lista = lista['label_names']
    for i, elem in enumerate(lista):
        if elem == label:
            return i


def get_some_class(list_files, class_name):
    """
    Return exemples of a specific class
    :param list_files: list of Strings that contains the names of cifar data files.
    :param class_name: String the name of desired class
    :return: List of examples and target label that are from this class
    """
    final_array = []
    for file in list_files:
        dict = unpickle('../cifar-10/' + file)
        example_list = dic_to_array(dict)
        for example in example_list:
            if example[1] == get_index_from_name_label(class_name):
                final_array.append(example)
    return final_array


def draw_img(img):
    """
    Draw an image
    :param img: numpy array
    """
    plt.imshow(img)
    plt.show()


def convolve_image(img, filter):
    """
    Apply a filter over a image

    :param img: numpy array representing original image
    :param filter: numpy array representing the filter that will be apply
    :return: img: numpy array with a filtered image
    """
    return cv2.filter2D(img, -1, filter)


def unflatten(img_flatten):
    """
    Transform a flattened image in its matrix form with rgb channels
    :param img_flatten: numpy ndarray with shape (3072,) it is one image
    :return: img numpy ndarray with shape (32, 32, 3) that is the image in matrix representation
    """
    img = img_flatten.reshape(3, 1024)
    r, g, b = img[0], img[1], img[2]
    r = r.reshape(32, 32)
    g = g.reshape(32, 32)
    b = b.reshape(32, 32)
    img = cv2.merge((b, g, r))
    return img
