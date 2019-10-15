#!/usr/bin/env python3

import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# $y = \sin(2\pi x)$
def real_func(x):
    return np.sin(2 * np.pi * x)

# $h(x) = \theta_0 \times x^2 + \theta_1 \times x^1 + \theta_2 \times x^0$
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# polynomial fitting, overfit
# target: $\min\sum_{i=1}^{n}(h(x_i) - y_i)^2$

# $h(x) - y$
def residuals_func(p, x, y):
    return fit_func(p, x) - y

# regularization
# target: $min\sum_{i=1}^{n}(h(x_i) - y_i)^2 + \lambda\Vert w \Vert^2$

lamb = 0.0001
def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5 * lamb * np.square(p)))
    return ret

x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)

y_r = real_func(x)
y = [np.random.normal(0, 0.1) + y for y in y_r]

def fitting(res_func, M, ax):
    p_init = np.random.rand(M + 1)
    p_lsq = leastsq(res_func, p_init, args=(x, y))
    print('Fitting parameters: ', p_lsq[0])

    ax.plot(x_points, real_func(x_points), label='real')
    ax.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    ax.plot(x, y, 'bo', label='noise')
    ax.legend()
    return p_lsq

if __name__ == '__main__':
    fig_poly = plt.figure()
    fig_poly.suptitle('Polynomial Fitting')
    for M in range(10):
        ax = fig_poly.add_subplot(2, 5, M + 1)
        p_lsq = fitting(residuals_func, M, ax)
    # plt.show()

    fig_reg = plt.figure()
    fig_reg.suptitle('Regularization')
    for M in range(10):
        ax = fig_reg.add_subplot(2, 5, M + 1)
        p_lsq = fitting(residuals_func_regularization, M, ax)
    plt.show()

'''
Ref: https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC01%E7%AB%A0%20%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E6%A6%82%E8%AE%BA/1.Introduction_to_statistical_learning_methods.ipynb
'''