from datasets import \
    X1_train, X1_test, y1_train, y1_test, \
    X2_train, X2_test, y2_train, y2_test, \
    X3_train, X3_test, y3_train, y3_test
from sklearn.neighbors import KNeighborsClassifier

# CREATE KNN CLASSIFIER
clf = KNeighborsClassifier(n_neighbors=5)

# TRAIN AND TEST CLASSIFIER
clf.fit(X1_train, y1_train)
print("Accuracy (shape descriptor):", clf.score(X1_test, y1_test))

clf.fit(X2_train, y2_train)
print("Accuracy (texture histogram):", clf.score(X2_test, y2_test))

clf.fit(X3_train, y3_train)
print("Accuracy (fine scale margin):", clf.score(X3_test, y3_test))