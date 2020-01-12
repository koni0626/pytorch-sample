# coding:UTF-8
import numpy as np
import torch
from torch import nn
from torch import optim
from torchsummary import summary

device = "cuda:0"

# ネットワークの読み込み
net = torch.load("sample-1.pth")

# ネットワークの表示
summary(net, (2, ))

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                   [-1, 32]              96
              ReLU-2                   [-1, 32]               0
            Linear-3                    [-1, 1]              33
              ReLU-4                    [-1, 1]               0
================================================================
"""

# XOR野データを作成
X = [[0., 0.],
     [0., 1.],
     [1., 0.],
     [1., 1.]]

# データをGPU上に作成
X = torch.tensor(X, device=device, dtype=torch.float32)

# 予測
y_pred = net(X)
print("予測結果")
print(y_pred)

# ReLU-4を外してSigmoidに変えてみる

# まず任意の層にアクセスする
for n in net:
    print(n)
    
# 4番目が出力層

net[3] = nn.Sigmoid()
for n in net:
    print(n)


# 予測 未学習のsigmoidになったので予測結果が変わる。
y_pred = net(X)
print("予測結果")
print(y_pred)
