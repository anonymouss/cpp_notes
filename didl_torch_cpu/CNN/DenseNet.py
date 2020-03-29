import torch
from torch import nn, optim
import torch.nn.functional as F

import sys, time
sys.path.append('..')
from helper import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk

net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
)

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    db = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module('DenseBlock_%d' % i, db)
    num_channels = db.out_channels
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module('transition_block_%d' % i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net.add_module('BN', nn.BatchNorm2d(num_channels))
net.add_module('relu', nn.ReLU())
net.add_module('global_avg_pool', GlobalAvgPool2d())
net.add_module('fc', nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))

X = torch.rand((1, 1, 96, 96))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)

batch_size, lr, num_epochs = 256, 0.001, 5
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)