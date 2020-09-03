import numpy as np
import matplotlib.pyplot as plt
from Kmeans import Kmeans

means = [[2, 2], [8, 3], [3, 6]] # các tâm dự tính :))
cov = [[1, 0], [0, 1]]           # ma trận hiệp phương sai
N = 500                          # tổng số điểm là 3 * N
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis=0)
K = 3

plt.plot(X[:, 0], X[:, 1], "ro", markersize=2)
plt.title("data set")
plt.show()


model = Kmeans(X, K)
(centers, labels) = model.train()

X0 = X[labels == 0, :]
X1 = X[labels == 1, :]
X2 = X[labels == 2, :]

plt.plot(X0[:, 0], X0[:, 1], "yo", markersize=2)
plt.plot(X1[:, 0], X1[:, 1], "go", markersize=2)
plt.plot(X2[:, 0], X2[:, 1], "bo", markersize=2)
plt.plot(centers[:, 0], centers[:, 1], "ro")
plt.title("3-means cluster")
plt.show()

