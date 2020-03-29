import torch
from torch import nn, optim
import torch.nn.functional as F

import time, sys
sys.path.append('..')
from helper import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4):
        super(Inception, self).__init__()
        self.path1_1 = nn.Conv2d(in_channels, out_channels1, kernel_size=1)
        self.path2_1 = nn.Conv2d(in_channels, out_channels2[0], kernel_size=1)
        self.path2_2 = nn.Conv2d(out_channels2[0], out_channels2[1], kernel_size=3, padding=1)
        self.path3_1 = nn.Conv2d(in_channels, out_channels3[0], kernel_size=1)
        self.path3_2 = nn.Conv2d(out_channels3[0], out_channels3[1], kernel_size=5, padding=2)
        self.path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_2 = nn.Conv2d(in_channels, out_channels4, kernel_size=1)

    def forward(self, X):
        p1 = F.relu(self.path1_1(X))
        p2 = F.relu(self.path2_2(F.relu(self.path2_1(X))))
        p3 = F.relu(self.path3_2(F.relu(self.path3_1(X))))
        p4 = F.relu(self.path4_2(self.path4_1(X)))
        return torch.cat((p1, p2, p3, p4), dim=1)

blk1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

blk2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

blk3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

blk4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

blk5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    GlobalAvgPool2d()
)

net = nn.Sequential(blk1, blk2, blk3, blk4, blk5, FlattenLayer(), nn.Linear(1024, 10))

'''
X = torch.rand(1, 1, 96, 96)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'out shape: ', X.shape)
'''

batch_size, lr, num_epochs = 128, 0.001, 5

train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)