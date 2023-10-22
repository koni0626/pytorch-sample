# coding:UTF-8
import torch
import torch.nn as nn

device = "cuda:0"

vocab_size = 10  # 例えば、10個の異なる単語/文字があるとします。
embedding_dim = 5  # 各単語/文字を5次元のベクトルに埋め込みたいとします。

embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
input_sequence = torch.tensor([1, 4, 7]).to(device)

embedded_sequence = embedding(input_sequence)
print(embedded_sequence)
embedded_sequence = embedding(input_sequence).view((1, 1, -1))
print(embedded_sequence)