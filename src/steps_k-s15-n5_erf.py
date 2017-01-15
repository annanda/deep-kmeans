# coding=utf-8
from steps_utils import *


def main():
    print("<READING CIFAR 10>")
    train, test = read_cifar10()
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

# RESULTS:
# 10%
