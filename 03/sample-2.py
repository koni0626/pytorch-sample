# coding:UTF-8
import numpy as np
import torch
from torch import nn

net1 = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU()
        )

net2 = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32)
        )
        

# XORのデータを作成
X = [[0., 0.],
     [0., 1.],
     [1., 0.],
     [1., 1.]]
     
X = torch.tensor(X)

pred_1 = net1(X)
print(pred_1)
pred_2 = net2(X)
print(pred_2)

print(net1[0].weight)
print(net2[2].weight)
print(net2[2].bias)
