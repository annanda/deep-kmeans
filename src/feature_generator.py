from sklearn.cluster import MiniBatchKMeans

from image_utils import *


def generate_centroids_from_dataset(data_set, centroids_size, num_centroids):
    """generate a set of centroids from image

    :param data_set: a numpy array of images (numpy array)
    :param centroids_size: int that will determinate a tuple (width, height) with the sample size
    :param num_centroids: the number of centroids that will be generated
    :return: centroids: numpy array with the k-means generated centroids
    """

    images = []
    for data in data_set:
        flat_image = data[0]
        image = flat_image.reshape(32, 32, 3)
        images.append(image)

    # GENERATING PATCHES
    patches = get_random_patches_of_images(images, patch_width=centroids_size, patch_height=centroids_size,
                                           num_patches_per_image=10)

    data_to_fit = []
    for patch in patches:
        flat_patch = patch.flatten()
        data_to_fit.append(flat_patch)

    # FITTING K-MEANS
    kmeans_model = MiniBatchKMeans(n_clusters=num_centroids, batch_size=100, n_init=10, max_no_improvement=10,
                                   verbose=True)
    kmeans_model.fit(data_to_fit)

    centroids = kmeans_model.cluster_centers_

    return centroids


def run():

    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_files = ['test_batch']

    train_set, valid_set, test_set = read_images_from_cifar_10(data_files, test_files)

    # configure aqui!
    centroids_size = 15
    num_centroids = 5

    # creating centroids from cifar 10
    centroids = generate_centroids_from_dataset(train_set, centroids_size, num_centroids)

    # drawing the grid with centroids
    images = []
    for centroid in centroids:
        image = centroid.reshape(centroids_size, centroids_size, 3)
        images.append(image)

        draw_image(image, "Centroid")

    # filtering and drawing a image
        # filtered = convolve_image(train_set[0][0].reshape(28, 28), centroids[0].reshape(5, 5))
        # draw_image(filtered)


if __name__ == '__main__':
    run()
