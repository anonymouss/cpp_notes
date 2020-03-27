import torch
from torch import nn
from collections import OrderedDict

br = '=' * 80

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        a = self.act(self.hidden(X))
        return self.output(a)

X = torch.rand(2, 784)
net = MLP()
print(net)
print(net(X))

print(br)

class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

print(net)
print(net(X))

print(br)

net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
print(net[-1])
print(net)

print(br)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

net = MyModule()
print(net)
net(torch.rand(2, 10))

print(br)

class Module_ModuleList(nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(10, 10)])

class Module_List(nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        self.fc = [nn.Linear(10, 10)]

net1 = Module_ModuleList()
net2 = Module_List()

print('net1:')
print(net1)
for p in net1.parameters():
    print(p.size())

print('net2:')
print(net2)
for p in net2.parameters():
    print(p.size())

print(br)

net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10)
print(net['linear'])
print(net.output)
print(net)

print(br)

class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        x = self.linear(x)
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))

print(br)

class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    
    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

net = nn.Sequential(nn.Linear(8, 4), CenteredLayer())
y = net(torch.rand(4, 8))
print(y.mean().item())

print(br)

# save/load model
# state_dict
class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP2()
print(net)
print(net.state_dict())

# 只有具有可学习参数的层（卷积，线性等）才有 state_dict
# optim 也有
optimizer = torch.optim.Adam(net.parameters())
print(optimizer.state_dict())

# 保存模型有两种方式，1. 只保存模型参数(state_dict)，2. 保存整个模型
MODEL_PATH = '../data_tmp/model_test/model_test.pth'
torch.save(net.state_dict(), MODEL_PATH)
model = MLP2()
model.load_state_dict(torch.load(MODEL_PATH))
print(model.state_dict())

print(br)

# GPU
# !nvidia-smi 查看显卡信息