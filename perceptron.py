import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data[:100,:2]
y = iris.target[:100] * 2 - 1

class Perceptron:
    """
    単純パーセプトロン
    eta : 学習率
    """

    def __init__(self, eta):
        self.eta = eta

    def learn(self, X, y):
        self.X_ = np.c_[np.ones(X.shape[0]), X]
        self.w_ = np.zeros(X.shape[1]+1)

        accurate = 0
        while accurate < X.shape[0]:
            accurate = 0
            for i in np.random.permutation(range(X.shape[0])):
                if self.w_.dot(self.X_[i,]) * y[i] > 0:
                    accurate += 1
                else:
                    self.w_ += self.eta * y[i] * self.X_[i,].T
        return self.w_

p = Perceptron(eta=0.1)

w = p.learn(X, y)

plt.scatter(X[:50,0], X[:50,1])
plt.scatter(X[50:,0], X[50:,1])
x1 = linspace(4.0, 7.5, 1000)
x2 = -(w[0] + w[1] * x1) / w[2]
plt.plot(x1, x2)
plt.show()
