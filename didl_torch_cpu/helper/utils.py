import torch
import torchvision
import torchvision.transforms as transforms

from IPython import display
from matplotlib import pyplot as plt
import random
import time

num_workers = 0

def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker',
                    'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_data_fashion_mnist(batch_size):
    # 60000 samples
    mnist_train = torchvision.datasets.FashionMNIST('../data_tmp/MNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    # 10000 samples
    mnist_test = torchvision.datasets.FashionMNIST('../data_tmp/MNIST', train=False, download=True,
                                                    transform=transforms.ToTensor())

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers)
    return train_iter, test_iter

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def sgd(params, lr, batch_size):
    for param in params:
        # 使用 .data 不会影响梯度传播
        param.data -= lr * param.grad / batch_size

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None,
            optimizer=None):
    for epoch in range(num_epochs):
        prev_ts = time.time()
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is not None:
                optimizer.step()
            else:
                sgd(params, lr, batch_size)

            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        train_ts = time.time()
        test_acc = evaluate_accuracy(test_iter, net)
        test_ts = time.time()
        print('Epoch {}, loss: {:4f}, train acc: {:3f} [train time: {}s],'\
            'test acc: {:3f} [test time: {}s]'.format(
            epoch + 1, train_loss_sum / n, train_acc_sum / n, train_ts - prev_ts,
            test_acc, test_ts - train_ts
        ))
        prev_ts = test_ts