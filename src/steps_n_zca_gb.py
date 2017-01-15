# coding=utf-8
from image_utils import *
from steps_utils import *


def main():
    print("<READING CIFAR 10>")
    train, test = read_some_labels_from_cifar10(wanted_labels=[LABEL_CAT, LABEL_BIRD])
    print("<NORMALIZATION>")
    normalize_set(train)
    normalize_set(test)
    print("<WHITENING>")
    whiten_set_with_zca(train)
    whiten_set_with_zca(test)
    print("<CLASSIFICATION WITH GRADIENT BOOSTING>")
    result = classify_with_gradient_boosting(train, test)
    print("<RESULT>")
    print(result)


if __name__ == "__main__":
    main()


# RESULTS (PER LABEL SET):
# CAT & BIRD
# 0.735
