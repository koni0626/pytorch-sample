# coding:UTF-8

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

device = "cuda:0"

"""
sample1の多層パーセプトロンをクラス化した場合のサンプル
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

# データをGPU上に作成
X = torch.tensor(X, device=device, dtype=torch.float32)
Y = torch.tensor(Y, device=device, dtype=torch.float32)

# ネットワークをGPUに転送
#net = net.to(device)
model.to(device)


# optimizerの作成(SGD)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()

# loss関数(最小二乗法)
loss_fn = nn.MSELoss()

"""
一回だけ予測→損失計算→バックプロパゲーションを行う場合
"""
# 予測
y_pred = model(X)
print("予測結果")
print(y_pred)

# 損失計算
loss = loss_fn(y_pred, Y)
print(loss)

# バックプロパゲーション
loss.backward()

# step関数がないと、バックプロパゲーションの結果が反映されない
optimizer.step()

# バックプロパゲーション後、もう一度予測
optimizer.zero_grad()
y_pred = model(X)
print("予測結果")
print(y_pred)


# 100回トレーニングする
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(X)
    print("result:{}".format(y_pred))
    
    loss = loss_fn(y_pred, Y)
    loss.backward()
    print("loss:{:.3f}".format(loss))
    optimizer.step()

# トレーニングしたデータを保存する
torch.save(model.state_dict(), "sample-5.pth")


