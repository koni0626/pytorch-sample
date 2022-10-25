# coding:UTF-8

import numpy as np
import torch
from torch import nn
from torch import optim

device = "cuda:0"

"""
多層パーセプトロン カテゴリ版
"""
net = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
        nn.Softmax()
    )

# XORのデータを作成
X = [[0., 0.],
     [0., 1.],
     [1., 0.],
     [1., 1.]]

# 教師データ。kerasの場合は[[1, 0], [0,1], [0, 1], [1, 0]]のように
# 指定するが、pytorchの場合はインデックス番号を入れるだけでよい
Y = [0,1,1,0]
     

# データをGPU上に作成
X = torch.tensor(X, device=device, dtype=torch.float32)
Y = torch.tensor(Y, device=device, dtype=torch.int64)

# ネットワークをGPUに転送
net = net.to(device)

# optimizerの作成(SGD)
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#adamのほうが収束が早い
optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer.zero_grad()

# loss関数 クロスエントロピー
loss_fn = nn.CrossEntropyLoss()

"""
一回だけ予測→損失計算→バックプロパゲーションを行う場合
"""
# 予測
#net.train()
y_pred = net(X)
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
y_pred = net(X)
print("予測結果")
print(y_pred)


# 100回トレーニングする
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = net(X)
    print("result:{}".format(y_pred))
    
    loss = loss_fn(y_pred, Y)
    loss.backward()
    print("loss:{:.3f}".format(loss))
    optimizer.step()


