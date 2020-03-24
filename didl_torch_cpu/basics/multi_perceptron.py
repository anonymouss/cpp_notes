import torch
from torch import nn

import numpy as np

import sys
sys.path.append('..')
from helper import *

'''
    ReLU:
        ReLU(x) = \max(x, 0)
    Sigmoid:
        sigmoid(x) = \frac{1}{1 + \exp(-x)}
        {sigmoid}'(x) = sigmoid(x)\left(1-sigmoid(x)\right )
    Tanh:
        tanh(x) = \frac{1-\exp(-2x)}{1+exp(-2x)}
        tanh'(x) = 1 - tanh^2(x)
'''

def xy_plot(x_vals, y_vals, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()

# some activation functions
# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = x.relu()
# xy_plot(x, y, 'ReLU')
# y.sum().backward()
# xy_plot(x, x.grad, 'ReLu\'')
# y = x.sigmoid()
# xy_plot(x, y, 'sigmoid')
# if x.grad is not None:
#     x.grad.zero_()
# y.sum().backward()
# xy_plot(x, x.grad, 'sigmoid\'')
# y = x.tanh()
# xy_plot(x, y, 'tanh')
# if x.grad is not None:
#     x.grad.data.zero_()
# y.sum().backward()
# xy_plot(x, x.grad, 'tanh\'')

####################################################################################################

# raw impl
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_epochs, lr = 5, 100.0

num_inputs, num_outputs, num_hiddens = 28 * 28, 10, 256

W1_r = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1_r = torch.zeros(num_hiddens, dtype=torch.float)
W2_r = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2_r = torch.zeros(num_outputs, dtype=torch.float)

params = [W1_r, b1_r, W2_r, b2_r]
for param in params:
    param.requires_grad_(True)

def relu(X):
    return torch.max(input=X, other=torch.zeros((len(X), 1), dtype=torch.float))

def multi_perceptron(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1_r) + b1_r)
    return torch.matmul(H, W2_r) + b2_r

loss = torch.nn.CrossEntropyLoss()
net = multi_perceptron
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])

####################################################################################################

# simple impl
class MultiPerceptron(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(MultiPerceptron, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, X):
        h = torch.relu(self.fc1(X.view((-1, num_inputs))))
        o = self.fc2(h)
        return o

net = MultiPerceptron(num_inputs, num_outputs, num_hiddens)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])