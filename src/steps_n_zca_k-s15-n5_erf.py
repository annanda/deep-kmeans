# coding=utf-8
from image_utils import *
from steps_utils import *


def main():
    cache_n_zca = "../n_zca.npy"
    if os.path.exists(cache_n_zca):
        print("<LOADING FROM CACHE '{}'>".format(cache_n_zca))
        cached = np.load(cache_n_zca)
        train = Dataset(list(cached[0][0]), list(cached[0][1]))
        test = Dataset(list(cached[1][0]), list(cached[1][1]))
    else:
        print("<READING CIFAR 10>")
        train, test = read_cifar10()
        print("<NORMALIZATION>")
        normalize_set(train)
        normalize_set(test)
        print("<WHITENING>")
        whiten_set_with_pca(train)
        whiten_set_with_pca(test)
        print("<WHITENING -- SAVING>")
        np.save(cache_n_zca, (train, test))

    print("<KMEANS>")
    centroids_size = 15
    num_centroids = 5
    centroids = generate_centroids_from_dataset(train, centroids_size, num_centroids)
    print("<CLASSIFICATION WITH EMSEMBLE OF RANDOM FOREST>")
    centroid_images = [centroid.reshape(15, 15, 3) for centroid in centroids]
    result = classify_with_ensemble_of_random_forest(train, test, centroid_images)
    print("<RESULT>")
    print(result)


if __name__ == "__main__":
    main()

# RESULTS
# 37%
