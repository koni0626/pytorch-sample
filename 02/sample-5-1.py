# coding:UTF-8

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

device = "cuda:0"

"""
sample5で学習したファイルを読み込んで実行する場合
"""

"""
多層パーセプトロン MSE版
"""
"""
net = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.ReLU()
    )
"""

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 1)
    
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
        
    
# XORのデータを作成
X = [[0., 0.],
     [0., 1.],
     [1., 0.],
     [1., 1.]]

Y = [[0.],
     [1.],
     [1.],
     [0.]]

model = MyModel() #ここ改定
model.load_state_dict(torch.load("sample-5.pth"), strict=False)

# データをGPU上に作成
X = torch.tensor(X, device=device, dtype=torch.float32)
Y = torch.tensor(Y, device=device, dtype=torch.float32)

# ネットワークをGPUに転送
#net = net.to(device)
model.to(device)




"""
一回だけ予測→損失計算→バックプロパゲーションを行う場合
"""
# 予測
y_pred = model(X)
print("予測結果")
print(y_pred)

