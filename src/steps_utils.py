# coding=utf-8
from collections import namedtuple, Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from image_utils import *

Dataset = namedtuple('Dataset', 'x y')


def read_some_labels_from_cifar10(wanted_labels=[]):
    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_file = 'test_batch'

    def add_file_data_to_dataset(file_name, dataset):
        file_data = unpickle('../cifar-10/' + file_name)
        for i in range(len(file_data['data'])):
            if file_data['labels'][i] in wanted_labels:
                dataset.x.append(file_data['data'][i])
                dataset.y.append(file_data['labels'][i])

    train_set = Dataset([], [])
    for file_name in data_files:
        add_file_data_to_dataset(file_name, train_set)

    test_set = Dataset([], [])
    add_file_data_to_dataset(test_file, test_set)

    return train_set, test_set


def read_cifar10():
    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_file = 'test_batch'

    train_set = Dataset([], [])
    for file_name in data_files:
        file_data = unpickle('../cifar-10/' + file_name)
        train_set.x.extend(file_data['data'])
        train_set.y.extend(file_data['labels'])

    file_data = unpickle('../cifar-10/' + test_file)
    test_set = Dataset(list(file_data['data']), file_data['labels'])

    return train_set, test_set


def normalize_set(dataset):
    for i in range(len(dataset.x)):
        flat_image = dataset.x[i]
        image = unflatten(flat_image)
        img_normalized = normalize_img(image)
        dataset.x[i] = flatten_img(img_normalized)


def whiten_set_with_zca(dataset):
    for i in range(len(dataset.x)):
        flat_image = dataset.x[i]
        image = unflatten(flat_image)
        img_whitened = whiten_image(image, is_zca=True)
        dataset.x[i] = flatten_img(img_whitened)


def whiten_set_with_pca(dataset):
    for i in range(len(dataset.x)):
        flat_image = dataset.x[i]
        image = unflatten(flat_image)
        img_whitened = whiten_image(image, is_zca=False)
        dataset.x[i] = flatten_img(img_whitened)


def convolve_set(dataset, centroid_image):
    convolved_dataset = Dataset([], dataset.y)
    for i in range(len(dataset.x)):
        image = dataset.x[i].reshape(32, 32, 3)
        channel = cv2.filter2D(image, -1, centroid_image)
        channel = channel.flatten()
        convolved_dataset.x.append(channel)

    return convolved_dataset


def classify_with_gradient_boosting(train, test):
    classifier = GradientBoostingClassifier()
    classifier.fit(train.x, train.y)
    predictions = classifier.predict(test.x)
    score = accuracy_score(test.y, predictions)

    return score


def classify_with_random_forest(train, test):
    classifier = RandomForestClassifier(n_jobs=4)
    classifier.fit(train.x, train.y)
    predictions = classifier.predict(test.x)
    score = accuracy_score(test.y, predictions)

    return score


def classify_with_ensemble_of_random_forest(train, test, centroid_images):
    predictions = []
    for i, centroid_image in enumerate(centroid_images):
        convolved_train = convolve_set(train, centroid_image)
        convolved_test = convolve_set(test, centroid_image)
        print("Classifing {}".format(i))
        classifier = RandomForestClassifier(n_jobs=4)
        classifier.fit(convolved_train.x, convolved_train.y)
        predictions.append(classifier.predict(convolved_test.x))

    voted_predictions = []
    for i in xrange(len(test.y)):
        predicted_classes = []
        for j in xrange(len(centroid_images)):
            predicted_classes.append(predictions[j][i])
        counter = Counter(predicted_classes)
        most_common_prediction = counter.most_common(1)[0][0]
        voted_predictions.append(most_common_prediction)

    score = accuracy_score(test.y, voted_predictions)

    return score


def main():
    # use para testar novos passos
    pass


if __name__ == "__main__":
    main()

'''
CODIGO ETAPAS:
c{\d\d*} - cifar só com classes \d. Ex: c23 é p cifar só com gatos e passaros. Se omitido, é o cifar todo
n - normalizado
pca/zca - whitening aplicado usando um dos dois metodos
k-s\d-c\d - kmeans e convolução, com tamanho dos centroids s e numero de centroids c
gb/rf/erf - classificação com gradient boosting ou random forest
'''
