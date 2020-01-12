# coding:UTF-8

import numpy as np
import torch
from torch import nn
from torch import optim
from torchsummary import summary

device = "cuda:0"

net = torch.load('sample-1.pth')
# XOR野データを作成
X = [[0., 0.]]

Y = [[0.],
     [1.],
     [1.],
     [0.]]
     

# データをGPU上に作成
X = torch.tensor(X, device=device, dtype=torch.float32)

# ネットワークの内容を表示する
net = net.to(device)
print(net)

# ネットワークの内容を表示する
summary(net, (2,))

# loss関数(最小二乗法)
loss_fn = nn.MSELoss()

# 予測
y_pred = net(X)
print("予測結果")
print(y_pred)

