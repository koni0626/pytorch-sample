# coding:UTF-8
import torch
import torch.nn as nn


input_size = 10  # 入力の特徴量の数
hidden_size = 10  # LSTMの隠れ状態のサイズ 全結合の重みみたいなもの
num_layers = 1  # LSTMの層の数

lstm = nn.LSTM(input_size, hidden_size, num_layers)

seq_len = 5  # シーケンスの長さ
batch_size = 1  # バッチのサイズ

# ランダムなテンソルを入力データとして生成
input_data = torch.randn(seq_len, batch_size, input_size)

# 初期の隠れ状態とセルの状態
h0 = torch.zeros(num_layers, batch_size, hidden_size)
c0 = torch.zeros(num_layers, batch_size, hidden_size)

# LSTMにデータを渡す
output, (hn, cn) = lstm(input_data, (h0, c0))
# htはoutputに含まれている
print(output)
print(hn)
print(cn)
