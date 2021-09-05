# coding:UTF-8
import numpy as np
import torch


"""
softmaxをpythonで書いてみる
"""
def softmax(record):
    data = np.array(record)
    data_max = np.max(data)
    data = data - data_max # 最大値で引いておかないとオーバーフローするらしい
    exp_a = np.exp(data)
    result = exp_a / np.sum(exp_a)
    return result

def softmax_overflow(record):
    data = np.array(record)
    exp_a = np.exp(data)
    result = exp_a / np.sum(exp_a)
    return result


if __name__ == '__main__':
    test_data = [1010., 1000., 990.]
    result = softmax(test_data)
    print(result)

    total = np.sum(result)
    print(total)


    result = softmax_overflow(test_data)
    print(result)
    total = np.sum(result)
    print(total)

    # overflowの原因はこれ
    # 指数に指定する値が大きすぎてnp.expがinfになってしまう。
    # だから最大値を引いておくのが良い。
    over_flow = np.exp(test_data)
    print(over_flow)

    # pytorchで書く場合はこんな感じ
    torch_soft_max = torch.nn.Softmax(dim=1)
    test_data = [test_data] #2次元配列に変換
    t_data = torch.tensor(test_data)
    result = torch_soft_max(t_data)
    print(result)


