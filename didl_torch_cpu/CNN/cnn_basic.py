import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((max(1, X.shape[0] - h + 1), max(1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1, -1]])
Y = corr2d(X, K)

conv2d = Conv2D(kernel_size=(1, 2))
step, lr = 50, 0.5
loss = nn.MSELoss()
optimizer = torch.optim.SGD(conv2d.parameters(), lr=lr)
for i in range(step):
    Y_hat = conv2d(X)
    l = loss(Y_hat, Y)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    print('Step {}, loss {:5f}'.format(i + 1, l.item()))

print(conv2d.state_dict())


def comp_conv2d(conv2d, X):
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
X = torch.rand(8, 8)
Y = comp_conv2d(conv2d, X)
print(Y)

def _conv2d(X, K, stride=None, padding=None):
    h, w = X.shape
    kh, kw = K.shape
    sh, sw, ph, pw = 0, 0, 0, 0
    if stride is not None:
        sh, sw = stride
    if padding is not None:
        ph, pw = padding
    oh = (h - kh + ph * 2) // sh + 1
    ow = (w - kw + pw * 2) // sw + 1
    ih = h + ph * 2
    iw = w + pw * 2
    print('input %dx%d, kernel %dx%d, stride (%d, %d), padding (%d, %d), output %dx%d' % (
        h, w, kh, kw, sh, sw, ph, pw, oh, ow
    ))
    P = torch.zeros(ih, iw)
    P[ph: ph + h, pw: pw + w] = X
    Y = torch.zeros(oh, ow)
    for i in range(oh):
        for j in range(ow):
            Y[i, j] = (P[i * sh: i * sh + kh, j * sw: j * sw + kw] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(_conv2d(X, K, [3, 2], [1, 1]))

def corr2d_multi_in(X, K):
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
print(corr2d_multi_in(X, K))

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])

K = torch.stack([K, K + 1, K + 2])
Y = corr2d_multi_in_out(X, K)
print(Y)
print(X.shape, Y.shape)

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)

X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().item())

def pool2d(X, pool_size, mode='max'):
    X = X.float()
    ph, pw = pool_size
    Y = torch.zeros(X.shape[0] - ph + 1, X.shape[1] - pw + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + ph, j: j + pw].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + ph, j: j + pw].mean()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))