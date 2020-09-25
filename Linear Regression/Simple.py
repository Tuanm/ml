import numpy as np
import matplotlib.pyplot as plt



class Simple:

	def __init__(self):
		self.x = None


	def fit(self, X, Y):
		"""
		Y = a0 + a1 X + epsi
		=> Y = [1 X] [a0 a1].T + epsi
		=> b =   A      x      + epsi
		=> |epsi| = |Ax - b|, |.| is Euclidean norm
		L(x) = (Ax - b)^2 => L'(x) = 2 A.T (Ax - b)
		L(x) minimized iff L'(x) = 0
		               iff A.T A x = A.T b
		"""
		one = np.ones((X.shape[0], 1))

		A = np.concatenate((one, X), axis=1)
		b = Y

		C = np.dot(A.T, A)
		d = np.dot(A.T, b)

		self.x = np.dot(np.linalg.pinv(C), d)



""" MAIN PROCEDURE BELOW """

def validate(X, Y, a0, a1):

	x0 = np.linspace(min(X), max(X))
	y0 = a0 + a1 * x0

	plt.plot(X.T, Y.T, 'ro')
	plt.plot(x0, y0)
	plt.show()



X = np.random.randint(100, size=(250, 1))
epsi = np.random.uniform(-25, 25, size=(250, 1))
Y = -9 + 6 * X + epsi

dataset = [X, Y]

X_train = dataset[0][0:200]
Y_train = dataset[1][0:200]
X_test = dataset[0][200:]
Y_test = dataset[1][200:]

model = Simple()
model.fit(X_train, Y_train)

a0, a1 = zip(model.x)

validate(X_test, Y_test, a0, a1)
