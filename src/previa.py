from image_utils import *


def run():
    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_files = ['test_batch']
    train_set, valid_set, test_set = read_images_from_cifar_10(data_files, test_files)
    img_2 = test_set[0][0]
    img_2 = unflatten(img_2)
    label_2 = test_set[0][1]
    # draw_image(img_2, 'image of test set do tipo {}'.format(label_2))
    matrix_image = get_random_patches_of_image(img_2, 8, 8, 4)
    draw_images_with_matplot_array_images(matrix_image)
    # img_part = get_random_patch_of_image(img_2, 16, 16)
    # draw_image(img_part, 'part of image of test set do tipo {}'.format(label_2))


if __name__ == '__main__':
    run()
