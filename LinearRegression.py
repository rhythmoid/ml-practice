import matplotlib.pyplot as plt
import numpy as np

class LinearReg:
    def __init__(self, eta=0.01, lambd=0.1, n_iter=10000):
        """
        線形回帰
        eta : 学習率η
        lambd : 正則化項の係数λ
        n_iter : 反復回数
        """
        self.eta = eta
        self.lambd = lambd
        self.n_iter = n_iter

    def learn(self, X, y):
        self.m_ , self.n_ = X.shape
        self.w_ = np.c_[np.random.randn(self.n_+1)]
        self.X_ = np.c_[np.ones(self.m_), X]
        self.y_ = np.c_[y]
        self.Jlist_ = [self.J(self.X_,y, self.w_)]
        for _iter in range(self.n_iter):
            self.w_ -= self.eta * self.grad(self.X_, self.y_, self.w_)
            self.Jlist_.append(self.J(self.X_, self.y_, self.w_))
        return self.Jlist_, self.w_

    def predict(self, X):
        return np.c_[np.ones(X.shape[0]), X].dot(self.w_)

    def J(self, X, y, w):
        """
        損失関数
        J = 1/(2m) * ( Σ(y - wx)^2 + λ||w||^2 )
        """
        return 1 / (2 * X.shape[0]) * (sum((X.dot(w) - y)**2) + self.lambd * (sum(w**2) - w[0]**2))

    def grad(self, X, y, w):
        """
        損失関数の微分
        ∂J/∂w_j = 1/m * ( Σ(y - wx)w_j + λw_j )
        """
        gr = 1 / X.shape[0] * (X.T.dot(X.dot(w) - y)  + self.lambd * w)
        gr[0] -= 1 / X.shape[0] * self.lambd * w[0]
        return gr

#データを生成
x = np.random.rand(100,1)
x = x * 2 - 1
y = 4 * x ** 3 + 2 * x - 1
y += np.random.randn(100,1)

#学習データ: 70個
x_train = x[:70]
y_train = y[:70]

#テストデータ: 30個
x_test = x[70:]
y_test = y[70:]

#9次式として回帰
X_train = np.c_[x_train ** 9, x_train ** 8, x_train ** 7,
                x_train ** 6, x_train ** 5, x_train ** 4,
                x_train ** 3, x_train ** 2, x_train]
X_test = np.c_[x_test ** 9, x_test ** 8, x_test ** 7,
                x_test ** 6, x_test ** 5, x_test ** 4,
                x_test ** 3, x_test ** 2, x_test]

#学習
l = LinearReg(eta=0.001, lambd=0.1, n_iter=100000)
J, w = l.learn(X_train, y_train)

#プロット
X_ = np.c_[np.ones(X_train.shape[0]), X_train]
plt.scatter(x_train, y_train, marker="+")
plt.scatter(x_train, X_.dot(w), marker="o")
plt.show()
plt.plot(J)
plt.show()

#決定係数を計算
y_mean = np.mean(y_test) * np.ones((len(y_test),1))
f = l.predict(X_test)
R2 = 1 - sum((y_test - f)**2)/sum((np.c_[y_test] - y_mean)**2)
R2 = R2[0]
print("coefficient of determination R2: {}".format(R2))
