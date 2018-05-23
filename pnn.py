from datasets import \
    X1_train, X1_test, y1_train, y1_test, \
    X2_train, X2_test, y2_train, y2_test, \
    X3_train, X3_test, y3_train, y3_test
from neupy import algorithms, environment
from sklearn import metrics

environment.reproducible()

# CREATE PNN CLASSIFIER
clf = algorithms.PNN()

# TRAIN AND TEST CLASSIFIER
clf.train(X1_train, y1_train)
y1_predicted = clf.predict(X1_test)
print("Accuracy (shape descriptor):", metrics.accuracy_score(y1_test, y1_predicted))

clf.train(X2_train, y2_train)
y2_predicted = clf.predict(X2_test)
print("Accuracy (texture histogram):", metrics.accuracy_score(y2_test, y2_predicted))

clf.train(X3_train, y3_train)
y3_predicted = clf.predict(X3_test)
print("Accuracy (fine scale margin):", metrics.accuracy_score(y3_test, y3_predicted))