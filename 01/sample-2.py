# coding:UTF-8
import numpy as np
import torch

# pytorchで使用するデータはtensorに変換する必要がある
data = [ [1, 2],
         [3, 4] ]

# リストをtensorに変換する
t_data = torch.tensor(data)
print(t_data)

# GPUのメモリに載せる
t_data = torch.tensor(data, device="cuda:0")
print(t_data)

# データのサイズを取得する
print(t_data.size())

# CPUにデータを戻す
c_data = t_data.to("cpu")
print(c_data)

# CPUに戻ったデータをnumpyに変換する
n_data = c_data.numpy()
print(n_data)


# 20×10の正規分布の乱数を作成する
t_r = torch.randn(20, 10).to("cuda:0")
print(t_r)

# 先頭[0:2, 0,5]だけ取得する

t_r = t_r[0:2, 0:5]
print(t_r)

# 平均値を求める
X = torch.randn(10)
print(X)
print(X.mean())
n_X = X.mean().numpy()
print(n_X)


# reshapeで2×2を4×1の行列に変換する
X = torch.randn(2, 2)
print(X)
f_X = X.view(4, 1)
print(f_X)

# 1×4に変換する
X = X.view(1, -1)
print(X)

# 行列の掛け算
X = torch.tensor([[1, 2], [3, 4]])
Y = torch.tensor([[5, 6], [7, 8]])
m = torch.mm(X, Y)
print(m)

# 内積(これは一次元どうし)
X = X.view(1, -1)
Y = Y.view(1, -1)
print(X)
m = torch.dot(X[0], Y[0])
print(m)

# 微分のテスト
# y=x^2を微分して2x(6)が得られるか
x = 3

x = torch.tensor(3., requires_grad=True)
y = x**2
y.backward()
#dy/dxだからxの傾きを知る。
print(x.grad)


