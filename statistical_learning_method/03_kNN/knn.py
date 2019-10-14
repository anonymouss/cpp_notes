#!/usr/bin/env python3

import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as sk_KNN
from collections import Counter, namedtuple
from operator import itemgetter

from time import process_time
from random import random


def L(x, y, p=2):
    # Lp distance:
    # $L_p(x_{i}, x_{j}) = \left \( \sum_{l=1}^n|x_i^{(l)} - x_j^{(l)}|^p \right \)^{\frac{1}{p}}$
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1.0 / p)
    else:
        return 0


def testLp():
    x = np.array([[1, 1], [5, 1], [4, 4]])
    for i in range(1, 5):
        distance = {'{}-{}'.format(x[0], y): L(x[0], y, p=i)
                    for y in [x[1], x[2]]}
        print(distance)


# prepare data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width',
              'petal length', 'petal width', 'label']
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def plot_data():
    plt.scatter(df[:50]['sepal length'], df[:50]
                ['sepal width'], label='class 0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]
                ['sepal width'], label='class 1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')


class MyKNN:
    def __init__(self, X_train, y_train, k_neighbors=3, p=2):
        self.k = k_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        knn_list = []
        for i in range(self.k):
            # Lp范数
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))
        for i in range(self.k, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            # 替换最大距离点，更新列表
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        return sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]

    def score(self, X_test, y_test):
        right_count = 0.0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1.0
        return right_count / len(X_test)


# kd - tree
class kdNode:
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k 维向量节点（k维空间中的一个样本点）
        self.split = split      # 整数（进行分割维度的序号）
        self.left = left        # 该结点分割超平面左子空间构成的 kd-tree
        self.right = right      # 该结点分割超平面右子空间构成的 kd-tree


result = namedtuple(
    'Result_tuples', 'nearest_point nearest_dist')


class kdTree:
    def __init__(self, data):
        k = len(data[0])
        if k > 0:
            self.k = k
        else:
            self.k = None
        self.nearest = None
        self.root = None

        def createTree_l(data, depth=0):
            if not data:
                return None
            k = len(data[0])
            axis = depth % k
            data.sort(key=itemgetter(axis))
            mid = len(data) // 2
            return kdNode(data[mid], axis, createTree_l(data[: mid], depth + 1),
                createTree_l(data[mid+1:], depth + 1))
        self.root = createTree_l(data)

    def findNearest(self, point, root=None, axis=0, dist_func=lambda x, y: L(x, y)):
        k = len(point)
        if root is None:
            return self.nearest

        # find leaf node
        if root.left or root.right:
            new_axis = (axis + 1) % self.k
            if point[axis] < root.dom_elt[axis] and root.left:
                self.findNearest(point, root.left, new_axis)
            elif root.right:
                self.findNearest(point, root.right, new_axis)

        # update nearest point
        dist = dist_func(root.dom_elt, point)
        if self.nearest is None or dist < self.nearest.nearest_dist:
            self.nearest = result(root.dom_elt, dist)

        # intersect with another area
        if abs(point[axis] - root.dom_elt[axis]) < self.nearest.nearest_dist:
            new_axis = (axis + 1) % self.k
            if root.left and point[axis] >= root.dom_elt[axis]:
                self.findNearest(point, root.left, new_axis)
            elif root.right and point[axis] < root.dom_elt[axis]:
                self.findNearest(point, root.right, new_axis)

        return self.nearest


def preorder_kdtree(root):
    print(root.dom_elt)
    if root.left:
        preorder_kdtree(root.left)
    if root.right:
        preorder_kdtree(root.right)


def random_vec(k):
    return [random() for _ in range(k)]


def random_mat(m, n):
    return [random_vec(n) for _ in range(m)]


if __name__ == '__main__':
    # fig = plt.figure()
    # fig.suptitle('kNN - kdTree')

    # testLp()

    # plot_data()

    # self defined knn
    knn_clf = MyKNN(X_train, y_train, 10)
    print('score [MyKNN]: ', knn_clf.score(X_test, y_test))

    # scikit-learn knn model
    # class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’,
    # algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None,
    # n_jobs=None, **kwargs)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    knn_sk = sk_KNN()
    knn_sk.fit(X_train, y_train)
    print('score [skKNN]: ', knn_sk.score(X_test, y_test))

    # kd tree
    data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    kdtree = kdTree(data)
    preorder_kdtree(kdtree.root)
    print(kdtree.findNearest([3, 4.5], kdtree.root))

    # kd tree test
    M = 400000
    start_time = process_time()
    kdtree2 = kdTree(random_mat(M, 3))
    res = kdtree2.findNearest([0.1,0.5,0.8], kdtree2.root)
    end_time = process_time()
    print('kdtree 2, time cost: ', end_time - start_time, 's')
    print('kdtree result: ', res)

    # plt.legend()
    # plt.show()

'''
Ref: https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC03%E7%AB%A0%20k%E8%BF%91%E9%82%BB%E6%B3%95/3.KNearestNeighbors.ipynb
'''
