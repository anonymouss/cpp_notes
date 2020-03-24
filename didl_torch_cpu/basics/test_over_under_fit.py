import torch

import numpy as np
import sys
sys.path.append('..')

from helper import *

# y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + \epsilon

n_train, n_test, w_true, b_true = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
labels = (w_true[0] * poly_features[:, 0] + w_true[1] * poly_features[:, 1]
        + w_true[2] * poly_features[:, 2] + b_true)
labels += torch.tensor(np.random.normal(0, 0.01, labels.shape), dtype=torch.float)

num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_loss, test_loss = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_loss.append(loss(net(train_features), train_labels).item())
        test_loss.append(loss(net(test_features), test_labels).item())
    print('Final epoch: train loss: {:5f}, test loss: {:5f}'.format(train_loss[-1], test_loss[-1]))
    semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss',
            range(1, num_epochs + 1), test_loss, ['train', 'test'])
    print('weight: ', net.weight.data, '\nbias: ', net.bias.data)

# normal fitting
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
            labels[:n_train], labels[n_train:])
# under fitting
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
            labels[n_train:])
# over fitting
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
             labels[n_train:])

# Approches to avoid overfitting
# 1. Weight decay/regularization
#   eg: \ell(w_1, w_2, b) = \frac{1}{n}\sum_{i=1}^{n}{\frac{1}{2}\left(x_1^{(i)}w_1+x_2^{(i)}w_2+b-y^{(i)}\right)^2}
#   let: \boldsymbol{w}=\left [w_1, w_2\right ]
#   new loss: \ell(w_1, w_2, b) + \frac{\lambda}{2n}\left\|\boldsymbol{w}\right\|^2

# test
# y = 0.05 + \sum_{i=1}^{p}{0.01x_i+\epsilon}
n_train, n_test, num_inputs = 20, 100, 200
w_true, b_true = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, w_true) + b_true
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train, :], labels[n_train:, :]

# raw impl
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return (w**2).sum() / 2

def linear_regression(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2

batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = linear_regression, squared_loss
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot2(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
            range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())

fit_and_plot2(lambd=0) # with no regularization
fit_and_plot2(lambd=5) # with regularization

# simple impl

def fit_and_plot_pytorch(wd):
    net = torch.nn.Linear(num_inputs, 1)
    torch.nn.init.normal_(net.weight, mean=0, std=1)
    torch.nn.init.constant_(net.bias, val=0)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
            range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())

fit_and_plot_pytorch(0)
fit_and_plot_pytorch(5)

# 2. dropout
# 隐藏层单元：h_i=\phi\left(x_1w_{1i} + x_2w_{2i} + x_3w_{3i} + x_4w_{4i} + b_i\right )
# 隐藏层单元有一定概率被丢弃，设丢弃概率为 p，则h_i有p的概率被清零，有(1-p)的概率会除以(1-p)拉伸

# raw impl
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float,
    requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float,
    requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float,
    requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]

drop_prob1, drop_prob2 = 0.2, 0.5

def dropout_net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:
        H1 = drop_out(H1, drop_prob1)
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = drop_out(H2, drop_prob2)
    return torch.matmul(H2, W3) + b3

# train
net = dropout_net
num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])

# simple impl
class DropoutNet(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, drop_prob1, drop_prob2):
        super(DropoutNet, self).__init__()
        self.dropout1 = torch.nn.Dropout(drop_prob1)
        self.dropout2 = torch.nn.Dropout(drop_prob2)
        self.fc1 = torch.nn.Linear(num_inputs, num_hiddens1)
        self.fc2 = torch.nn.Linear(num_hiddens1, num_hiddens2)
        self.fc3 = torch.nn.Linear(num_hiddens2, num_outputs)

    def forward(self, X):
        h1 = self.dropout1(self.fc1(X.view(X.shape[0], -1)).relu())
        h2 = self.dropout2(self.fc2(h1).relu())
        return self.fc3(h2)

net = DropoutNet(num_inputs, num_outputs, num_hiddens1, num_hiddens2, drop_prob1, drop_prob2)
torch.nn.init.normal_(net.fc1.weight, mean=0, std=0.01)
torch.nn.init.constant_(net.fc1.bias, val=0)
torch.nn.init.normal_(net.fc2.weight, mean=0, std=0.01)
torch.nn.init.constant_(net.fc2.bias, val=0)
torch.nn.init.normal_(net.fc3.weight, mean=0, std=0.01)
torch.nn.init.constant_(net.fc3.bias, val=0)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])