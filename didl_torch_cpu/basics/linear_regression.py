import torch

import numpy as np
import random
from time import time

import sys
sys.path.append('..')
from helper import *

'''
    house price prediction
    - x1:   house area
    - x2:   house age
    - y :   house price

    y_ = x_1 \times w_1 + x_2 \times w_2 + b

    cost function: MSE
    loss = \frac{1}{N}\sum_{i=1}^{N}{(\bar{y}_i - y_i)^2}

    w_1^*w_2^*b^* = \arg{\min_{w_1, w_2,b}{\operatorname{l}(w_1, w_2, b))}}

    optim: mini-batch SGD
'''

num_inputs = 2
num_examples = 100
true_w = [2, -3.4]
true_b = 4.2

# generate training data
features = torch.randn(num_examples, num_inputs).float()
labels = torch.mm(features, torch.tensor(true_w).float().unsqueeze(1)) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size())).float()

set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy())
# plt.show()

####################################################################################################

# raw implementation
w_r = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1))).float()
# NOTE: 如果把 requires_grad=True 在转换为Tensor时就设置好，那 grad_fn就会是
#       CopyBackward. 因为后面做了 float()转换？
#       https://discuss.pytorch.org/t/none-grad-attribute-in-multiprocessing-autograd-with-floattensor-type/20482
b_r = torch.zeros(1).float()
w_r.requires_grad_(True)
b_r.requires_grad_(True)

def linear_regression(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2

def sgd(params, lr, batch_size):
    for param in params:
        # 使用 .data 不会影响梯度传播
        param.data -= lr * param.grad / batch_size

# train
lr = 0.05
num_epochs = 10
batch_size = 10
net = linear_regression
loss = squared_loss

print('Training with raw implemented linear regression model')
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w_r, b_r), y).sum() # / len(X)
        l.backward()
        sgd([w_r, b_r], lr, batch_size)

        w_r.grad.data.zero_()
        b_r.grad.data.zero_()

    train_loss = loss(net(features, w_r, b_r), labels).mean()
    print('Epoch {}, loss {:2f}'.format(epoch + 1, train_loss.item()))

print('True    w: {}, b: {}'.format(true_w, true_b))
print('Trained w: {}, b: {}'.format(w_r.data.squeeze().numpy(), b_r.data.squeeze().numpy()))


####################################################################################################

# simple implementation by torch.nn.Module
import torch.utils.data as Data
from torch import nn
from torch.nn import init
import torch.optim as optim

dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# model
class LinearNet(nn.Module):
    def __init__(self, n_features):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

# train
print('Training with nn.Module linear model')
net = LinearNet(num_inputs)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.05)

for epoch in range(10):
    for X, y in data_iter:
        out = net(X)
        l = loss(out, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('Epoch {}, loss {:2f}'.format(epoch + 1, l.item()))

print('True    w: {}, b: {}'.format(true_w, true_b))
print('Trained w: {}, b: {}'.format(net.linear.weight.data.squeeze().numpy(), net.linear.bias.data.squeeze().numpy()))
