#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot_data

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from collections import Counter
import math

def prepare_data(test_size=0.2):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width',
                  'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    return train_test_split(X, y, test_size=test_size)

X_train, X_test, y_train, y_test = prepare_data()

# 高斯判别器
# 高斯概率密度函数：$P(x_i|y_k) = \frac{1}{\sqrt{2\pi\sigma^2_{y_k}}}\exp\left(-\frac{(x_i-\mu_{y_k})^2}{2\sigma^2_{y_k}}\right)$
# 数学期望：$\mu$
# 方差：$\sigma^2=\frac{\sum{(X-\mu)^2}}{N}$
class NaiveBayes:
    def __init__(self):
        self.model = None

    def mean(self, X):
        return sum(X) / float(len(X))

    def std(self, X):
        mu = self.mean(X)
        return math.sqrt(sum([math.pow(x - mu, 2) for x in X]) / float(len(X)))

    def gaussian_probability(self, x, mu, std):
        exp = math.exp(-(math.pow(x - mu, 2) / 2 / math.pow(std, 2)))
        return exp / math.sqrt(2 * math.pi) / std

    def summarize(self, train_data):
        summaries = [(self.mean(i), self.std(i)) for i in zip(*train_data)]
        return summaries

    def train(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for x, label in zip(X, y):
            data[label].append(x)
        self.model = {
            label: self.summarize(value) for label, value in data.items()
        }
        print('gaussian naive bayes training done!')

    def calculate_probabilities(self, x):
        probilities = {}
        for label, param in self.model.items():
            probilities[label] = 1
            for i in range(len(param)):
                mu, std = param[i]
                probilities[label] *= self.gaussian_probability(x[i], mu, std)
        return probilities

    def predict(self, X_test):
        return sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            h = self.predict(X)
            if h == y:
                right += 1
        acc = right / float(len(X_test))
        print('accuracy: ', acc)
        return acc


if __name__ == '__main__':
    nb_model = NaiveBayes()
    nb_model.train(X_train, y_train)
    nb_model.score(X_test, y_test)

    # scikit-learn
    sk_model = GaussianNB()
    sk_model.fit(X_train, y_train)
    print('scikit-learn, gaussian: ', sk_model.score(X_test, y_test))

'''
ref:
https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC04%E7%AB%A0%20%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF/4.NaiveBayes.ipynb
https://www.cnblogs.com/yemanxiaozu/p/7680761.html
https://www.cnblogs.com/evenelina/p/8434437.html
'''