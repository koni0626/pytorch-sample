# coding: UTF-8
import torch
import torch.nn as nn

if __name__ == '__main__':
    net = nn.Sequential()

    net1 = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )

    net2 = nn.Sequential(
        nn.Linear(1, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )

    net.add_module("net1", net1)
    net.add_module("net2", net2)
    net.add_module('fc1', nn.Linear(1, 10))
    print(net)
