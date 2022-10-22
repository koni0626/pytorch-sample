# coding: UTF-8
import numpy as np
import torch
from torch import nn
from torch import optim


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        print(x)
        x = self.relu1(x)
        print(x)
        x = self.fc2(x)
        print(x)
        x = self.sigmoid(x)
        print(x)
        return x

    def disp_weight(self):
        print(self.fc1.weight)
        print(self.fc1.bias)



def main():
    net = MyNet()
    x = [
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
    ]

    y = [
            [0.],
            [1.],
            [1.],
            [0.]
    ]

    x = torch.tensor(x)
    pred = net(x)
    net.disp_weight()


if __name__ == '__main__':
    main()
