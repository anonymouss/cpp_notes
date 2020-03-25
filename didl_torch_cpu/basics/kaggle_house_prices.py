# https://www.kaggle.com/c/house-prices-advanced-regression-techniques

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os, sys
sys.path.append('..')
from helper import *

torch.set_default_tensor_type(torch.FloatTensor)
DATA_ROOT = '../data_tmp/kaggle/house_prices'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
SUBMISSION_NAME = 'submission.csv'

# prepare data
train_data = pd.read_csv(os.path.join(DATA_ROOT, TRAIN_FILE_NAME))
test_data = pd.read_csv(os.path.join(DATA_ROOT, TEST_FILE_NAME))

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values)
test_features = torch.tensor(all_features[n_train:].values)
train_labels = torch.tensor(train_data.iloc[:, -1].values).view(-1, 1)

# define model, linear regression model
class LinearModel(nn.Module):
    def __init__(self, num_features):
        super(LinearModel, self).__init__()
        self.num_features = num_features
        self.fc1 = nn.Linear(num_features, 1)

    def forward(self, X):
        return self.fc1(X.view(-1, self.num_features))

mse_loss = nn.MSELoss()

class LogRMSE(nn.Module):
    def __init__(self):
        super(LogRMSE, self).__init__()

    def forward(self, y_hat, y):
        y_hat_clipped = torch.max(y_hat, torch.tensor(1.0))
        return torch.sqrt(mse_loss(y_hat_clipped.log(), y.log()))

lrmse_loss = LogRMSE()

# train
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr,
    weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net = net.float()
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            y_hat = net(X.float()).float()
            l = mse_loss(y_hat, y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        with torch.no_grad():
            y_hat_train = net(train_features.float()).float()
            train_ls.append(lrmse_loss(y_hat_train, train_labels.float()))
            if test_labels is not None:
                y_hat_test = net(test_features.float()).float()
                test_ls.append(lrmse_loss(y_hat_test, test_labels.float()))
    return train_ls, test_ls

# k-fold eval
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx, :]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = LinearModel(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == (k - 1):
            semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'log rmse',
                    range(1, num_epochs + 1), valid_ls, ['train', 'valid'])
        print('Fold {}, train: lrmse {:5f}, valid lrmse: {:5f}'.format(
            i, train_ls[-1], valid_ls[-1]
        ))
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('{}-fold validation: avg traim lrmse: {:5f}, avg valid lrmse {:5f}'.format(k, train_l, valid_l))

# predict
def train_and_predict(train_features, test_features, train_labels, test_data, num_epochs, lr,
    weight_decay, batch_size):
    net = LinearModel(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay,
        batch_size)
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'lrmes')
    print('train lrms %f' % train_ls[-1])
    preds = net(test_features.float()).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv(os.path.join(DATA_ROOT, SUBMISSION_NAME), index=False)

train_and_predict(train_features, test_features, train_labels, test_data, num_epochs, lr,
    weight_decay, batch_size)