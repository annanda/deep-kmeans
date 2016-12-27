import cv2
import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt


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


def read_images_from_cifar_10(file_path, normalize=True, whitenning=True):
    """
    Read images from cifar 10 folder and prepare the train set, valid set and test set

    :param file_path: path for cifar 10 cpickled file
    :param normalize:
    :param whitenning:
    :return: train_set (8k examples): numpy ndarray shape(8000, 2). row[i][0] = [image], row[i][1] = label
    :return: valid_set (2k examples): numpy ndarray shape(2000, 2). row[i][0] = [image], row[i][1] = label
    :return: test_set (2k examples): numpy ndarray shape(2000, 2). row[i][0] = [image], row[i][1] = label
    """

    train_dict = unpickle(file_path + 'data_batch_1')

    test_dict = unpickle(file_path + 'test_batch')

    classe_1_train = train_dict['data'][:4000]
    classe_1_valid = train_dict['data'][4000:5000]
    classe_2_train = train_dict['data'][5000:9000]
    classe_2_valid = train_dict['data'][9000:]

    classe_1_train_label = train_dict['labels'][:4000]
    classe_1_valid_label = train_dict['labels'][4000:5000]
    classe_2_train_label = train_dict['labels'][5000:9000]
    classe_2_valid_label = train_dict['labels'][9000:]

    train = np.append(classe_1_train, classe_2_train, axis=0)
    train_label = np.append(classe_1_train_label, classe_2_train_label, axis=0)
    # make association between each image with its label
    train_and_label = zip(train, train_label)
    train_set = np.array(list(train_and_label))

    valid_set = np.append(classe_1_valid, classe_2_valid, axis=0)
    valid_label = np.append(classe_1_valid_label, classe_2_valid_label, axis=0)
    # make association between each image with its label
    valid_set = zip(valid_set, valid_label)
    valid_set = np.array(list(valid_set))

    test_dict['data'] = test_dict['data'][2000:4000]
    test_dict['labels'] = test_dict['labels'][2000:4000]
    test_set = dic_to_array(test_dict)

    return train_set, valid_set, test_set

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
                subs[i][j].imshow(img.reshape(width, height))

            else:
                subs[i][j].imshow(img)

            subs[i][j].axis('off')

    plt.show()


def draw_img(img):
    """
    Draw an image

    :param img: numpy array
    """

    plt.imshow(img)
    plt.show()


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


def run():
    # train_set, valid_set, test_set = read_images_from_cifar_10('../cifar-10/')
    # img_1 = train_set[0][0]
    # target = train_set[0][1]
    # print target
    # img = un_flatten(img_1)
    # draw_img(img)
    print(get_label_names(0))

if __name__ == '__main__':
    run()