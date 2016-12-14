from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from image_utils import *


def generate_cluster_centroids(data_set, window_size, num_cluster):
    """generate a set of centroids from image

    :param data_set: a numpy array of images (numpy array)
    :param window_size: int that will determinate a tuple (width, height) with the sample size
    :param num_cluster: the number of centroids that will be generated
    :return: centroids: numpy array with the k-means generated centroids
    """

    X = np.array(data_set[0])
    y = data_set[1]

    sqrt_num_cluster = int(np.sqrt(num_cluster))

    #  GENERATING SAMPLES
    samples = generate_samples(X, 10000, window_size=(window_size,window_size))

    # FITTING K-MEANS
    kmeans_model = MiniBatchKMeans(n_clusters=num_cluster, batch_size=100, n_init=10, max_no_improvement=10, verbose=False)
    kmeans_model.fit(samples)

    centroids = kmeans_model.cluster_centers_

    return centroids


def run():

    train_set, valid_set, test_set = read_images_from_mnist("../mnist/mnist.pkl.gz")

    window_centroids_size = 5
    num_cluster = 16

    # creating centroids from mnist
    centroids = generate_cluster_centroids(train_set, window_centroids_size, num_cluster)

    # drawing the grid with centroids
    draw_multiple_images(centroids, 4, 4)

    # filtering and drawing a image
    filtered = convolve_image(train_set[0][0].reshape(28, 28), centroids[0].reshape(5, 5))
    draw_img(filtered)


if __name__ == '__main__':
    run()