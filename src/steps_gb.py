# coding=utf-8
from time import time

from steps_utils import *


def main():
    print(time())
    print("<READING CIFAR 10>")
    train, test = read_cifar10()
    print("<CLASSIFICATION WITH GRADIENT BOOSTING>")
    print(time())
    result = classify_with_gradient_boosting(train, test)
    print(time())
    print("<RESULT>")
    print(result)


if __name__ == "__main__":
    main()

# RESULTS (PER LABEL SET):
# ALL
# 48% (demorou 5 horas para rodar)
