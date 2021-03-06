# coding:UTF-8

import numpy as np
import torch
from torch import nn
from torch import optim

device = "cuda:0"

"""
多層パーセプトロン MSE版
"""
net = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.ReLU()
    )

# XOR野データを作成
X = [[0., 0.],
     [0., 1.],
     [1., 0.],
     [1., 1.]]

Y = [[0.],
     [1.],
     [1.],
     [0.]]
     

# データをGPU上に作成
X = torch.tensor(X, device=device, dtype=torch.float32)
Y = torch.tensor(Y, device=device, dtype=torch.float32)

# ネットワークをGPUに転送
net = net.to(device)

# optimizerの作成(SGD)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()

# loss関数(最小二乗法)
loss_fn = nn.MSELoss()

"""
一回だけ予測→損失計算→バックプロパゲーションを行う場合
"""

# 100回トレーニングする
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = net(X)
    print("result:{}".format(y_pred))
    
    loss = loss_fn(y_pred, Y)
    loss.backward()
    print("loss:{:.3f}".format(loss))
    optimizer.step()

# 入力層を取得
input_layer = net[0]
# 入力層のweightを表示
print(input_layer.weight)
# 入力層のバイアスを表示
print(input_layer.bias)
