import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')
from helper import *

'''
    softmax function:
    \hat{y_1},\hat{y_2},\hat{y_3} = \operatorname{softmax}(o1, o2, o3)
    \hat{y_i} = \frac{\exp(o_i)}{\sum_{j=1}^{N}{o_i}}

    loss: cross entropy
    H\left(y^{(i)},\hat{y}^{(i)}\right ) = -\sum_{j=1}^{q}{y_{j}^{(j)}\log{\hat{y}_{j}^{(j)}}}
    \ell\left(\Theta\right) = \frac{1}{n}\sum_{i=1}^{n}{H\left(y^{(i)}, \hat{y}^{(i)}\right)}
'''

batch_size = 256

train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10

####################################################################################################

# softmax raw implementation
W_r = torch.tensor(np.random.normal(0, 0.01,(num_inputs, num_outputs))).float()
b_r = torch.zeros(num_outputs).float()
W_r.requires_grad_(True)
b_r.requires_grad_(True)

def softmax(X):
    X_exp = X.exp()
    sum = X_exp.sum(dim=1, keepdim=True)
    return X_exp / sum

def softmax_regression(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W_r) + b_r)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


num_epochs, lr = 5, 0.1
net = softmax_regression

# train
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W_r, b_r], lr)

X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])

####################################################################################################

# simple impl
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y

net = LinearNet(num_inputs, num_outputs)

nn.init.normal_(net.linear.weight, mean=0, std=0.01)
nn.init.constant_(net.linear.bias, val=0)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])
