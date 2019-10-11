#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

# load data
iris = load_iris()
# print(iris)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df)
df['label'] = iris.target
# print(df)
df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]
# print(df)
counts = df.label.value_counts()
# print(counts)

def plot_data():
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='class 0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='class 1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    # plt.legend()
    # plt.show()

# prepare data
# fetch only two classes (iris dataset contains three classes). use first two features
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])

# perceptron
class MyPerceptron:
    def __init__(self):
        # w length equals to feature number
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.eta = 0.1
        self.alpha = np.zeros(len(X), dtype=np.float32)

    def reset(self):
        self.__init__()

    def sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def compute_gram(self, X):
        m = len(X)
        gram = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                gram[i][j] = X[i] * np.mat(X[j]).transpose()
        return gram

    def f(self, x, w, b):
        return np.dot(x, w) + b

    def f2(self, alpha, y, g, b):
        sum = 0
        for j in range(len(y)):
            sum += alpha[j] * y[j] * g[j]
        sum += b
        return sum

    # model:
    # $f(x) = sign(w\cdot x + b)$
    # $sign(x) = \left\{\begin{matrix} +1, x \ge 0 & \\-1, x < 0 & \end{matrix}\right.$
    # cost function:
    # $L(w,b) = -\sum_{x_i\subseteq M}{y_i(w\cdot x_i + b)}$
    # gradient descent updating:
    # $w = w + \eta y_ix_i$
    # $b = b + \eta y_i$
    def fit(self, X_train, y_train):
        has_wrong_classified = True
        while has_wrong_classified:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.f(X, self.w, self.b) <= 0:
                    self.w = self.w + self.eta * np.dot(y, X)
                    self.b = self.b + self.eta * y
                    wrong_count += 1
            if wrong_count == 0:
                has_wrong_classified = False
        print('Fitting done. w, b = ', self.w, self.b)

    # model:
    # $f(x) = sign\left \{ \sum_{j=1}^N{\alpha_j y_j x_j\cdot x+b} \right \}$
    # Miss condition:
    # $y_i\left \{ \sum_{j=1}^N{\alpha_jy_jx_j\cdot x_i + b} \right \} \le 0$
    # Gram Maxtrix:
    # $G = \left [ x_i\cdot x_j \right ]_{N\times N}$
    # Update:
    # $\alpha_i = \alpha_i + \eta$
    # $b = b + \eta y_i$
    # finally, $w = \sum_{i=1}^N{\alpha_i y_i x_i}$
    def fit2(self, X_train, y_train):
        self.eta = 1 # set $\eta$ to 1
        has_wrong_classified = True
        gram = self.compute_gram(X_train)
        while has_wrong_classified:
            wrong_count = 0
            for d in range(len(X_train)):
                g = gram[:, d]
                if y_train[d] * self.f2(self.alpha, y_train, g, self.b) <= 0:
                    self.alpha[d] += self.eta
                    self.b += y_train[d] * self.eta
                    wrong_count += 1
            if wrong_count == 0:
                has_wrong_classified = False
        for i in range(len(X_train)):
            self.w += self.alpha[i] * X_train[i] * y_train[i]
        print('Fitting 2 done. w, b = ', self.w, self.b)    


if __name__ == '__main__':
    fig_per = plt.figure()
    fig_per.suptitle('Perceptron')
    plot_data()
    # train
    perceptron = MyPerceptron()
    perceptron.fit(X, y)

    x_points = np.linspace(4, 7, 10)
    # w_0 * x + w_1 * h + b = 0 --> h = -(b + w_0 * x) / w_1
    h = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
    plt.plot(x_points, h, label='self impl perceptron')

    # perceptron in dual form
    perceptron.reset()
    perceptron.fit2(X, y)
    h2 = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
    plt.plot(x_points, h2, label='self impl perceptron, dual form')

    # sklearn perceptron 1
    clf1 = Perceptron(fit_intercept=True, max_iter=1000, shuffle=True)
    clf1.fit(X, y)
    print('w, h = ', clf1.coef_, clf1.intercept_)
    h_s1 = -(clf1.coef_[0][0] * x_points + clf1.intercept_) / clf1.coef_[0][1]
    plt.plot(x_points, h_s1, label='sklearn perceptrron 1, tol=default')

    # sklearn percetron 2. involve tol
    clf2 = Perceptron(fit_intercept=True, max_iter=1000, tol=None, shuffle=True)
    clf2.fit(X, y)
    print('w, h = ', clf2.coef_, clf2.intercept_)
    h_s2 = -(clf2.coef_[0][0] * x_points + clf2.intercept_) / clf2.coef_[0][1]
    plt.plot(x_points, h_s2, label='sklearn perceptrron 2, tol=None')

    plt.legend()
    plt.show()

'''
ref: https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC02%E7%AB%A0%20%E6%84%9F%E7%9F%A5%E6%9C%BA/2.Perceptron.ipynb
'''