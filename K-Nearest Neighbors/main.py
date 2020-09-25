import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X0 = iris_X[iris_y == 0, :]
X1 = iris_X[iris_y == 1, :]
X2 = iris_X[iris_y == 2, :]


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)


classifier = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)

print("predict:     ", y_predict[:20])
print("ground truth:", y_test[:20])
print("accuracy: %.2f%%" % (accuracy * 100))


classifier = neighbors.KNeighborsClassifier(n_neighbors=10, p=2) # weights='uniform'
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)

print("predict:     ", y_predict[:20])
print("ground truth:", y_test[:20])
print("accuracy: %.2f%%" % (accuracy * 100))


classifier = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)

print("predict:     ", y_predict[:20])
print("ground truth:", y_test[:20])
print("accuracy: %.2f%%" % (accuracy * 100))


def weight(distances):
	"""
	w_i = exp(-||x - x_i||^2 / sigma^2),
	||.|| is l2-norm
	"""
	sigma2 = 0.5 # changable
	return np.exp(-distances ** 2 / sigma2)

classifier = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights=weight)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)

print("predict:     ", y_predict[:20])
print("ground truth:", y_test[:20])
print("accuracy: %.2f%%" % (accuracy * 100))