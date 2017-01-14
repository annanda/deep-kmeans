import numpy as np
from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

datasets = np.load("../generated_features.npy")
X_train = []
y_train = []
X_validation = []
y_validation = []
X_test = []
y_test = []
for data in datasets[0]:
    X_train.append(data[0][0].flatten())
    y_train.append(data[1])
for data in datasets[1]:
    X_validation.append(data[0][0].flatten())
    y_validation.append(data[1])
for data in datasets[2]:
    X_test.append(data[0][0].flatten())
    y_test.append(data[1])


def try_classifier(clf):
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    score = accuracy_score(y_test, predictions)
    print("Accuracy score is: {}".format(score))

    predictions = cross_val_predict(clf, X_train, y_train, cv=5)
    print("Logloss: %0.2f (+/- %0.2f)" % (predictions.mean(), predictions.std() * 2))

    score = accuracy_score(y_test, predictions)
    print("Accuracy score is: {}".format(score))


try_classifier(GradientBoostingClassifier())
