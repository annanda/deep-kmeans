# coding=utf-8
from steps_utils import *


def main():
    print("<READING CIFAR 10>")
    train, test = read_cifar10()
    print("<CLASSIFICATION WITH GRADIENT BOOSTING>")
    result = classify_with_random_forest(train, test)
    print("<RESULT>")
    print(result)


if __name__ == "__main__":
    main()

# RESULTS:
# ALL
# 36%
