from image_utils import *


def run():
    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_files = ['test_batch']
    train_set, valid_set, test_set = read_images_from_cifar_10(data_files, test_files)
    img_2 = test_set[0][0]
    img_2 = unflatten(img_2)
    label_2 = test_set[0][1]
    draw_image(img_2, 'image of test set do tipo {}'.format(label_2))


if __name__ == '__main__':
    run()
