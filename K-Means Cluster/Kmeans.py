import numpy as np
from scipy.spatial.distance import cdist


class Kmeans:

	def __init__(self, X, K):
		self.X = X
		self.K = K

	def dist(self, X, centers):
		pass

	def init_centers(self):
		# chọn ngẫu nhiên K hàng của X để làm centers
		return self.X[np.random.choice(self.X.shape[0], self.K, replace=False)]

	def assign_labels(self, centers):
		# tính khoảng cách của từng cặp một
		D = cdist(self.X, centers)
		# trả về chỉ số của center gần nhất
		return np.argmin(D, axis=1)

	def update_centers(self, labels):
		centers = np.zeros((self.K, self.X.shape[1]))
		for k in range(self.K):
			# chọn tất cả điểm có nhãn k
			Xk = self.X[labels == k, :]
			# lấy trung bình các điểm để tìm center k
			centers[k, :] = np.mean(Xk, axis=0)
		return centers

	def has_converged(self, centers, new_centers):
		# nếu hai tập centers là giống nhau thì dừng
		return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

	def train(self):
		centers = self.init_centers()
		labels = []
		iters = 0
		while True:
			labels = self.assign_labels(centers)
			new_centers = self.update_centers(labels)
			if self.has_converged(centers, new_centers): break
			centers = new_centers
		return (centers, labels)