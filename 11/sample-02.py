# coding: UTF-8
import numpy as np
import torch
import torchvision.transforms.functional
from torch import nn
from torch import optim


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        middle_num = 4

        dummy_weight = [
            [1., 0.],
            [1., 0.],
            [1., 0.],
            [1., 0.],
        ]

        dummy_weight = torch.tensor(dummy_weight)
        self.dense = nn.Linear(2, middle_num)
        print(self.dense.weight)
        #torch.nn.init.ones_(self.dense.weight)
        #torch.nn.init.ones_(self.dense.bias)
        print(self.dense.state_dict())
        #self.dense.state_dict()["weight"] = dummy_weight
        print(self.dense.weight)
        #print(self.dense.weight)
        self.relu = nn.ReLU()
        self.out = nn.Linear(middle_num, 1)
        #torch.nn.init.ones_(self.out.weight)
        #torch.nn.init.ones_(self.out.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dense(x)
        print("↓↓↓↓↓↓↓↓↓↓")
        print(self.dense.weight)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    # XORのデータを作成
    X = [[0., 0.],
         [0., 1.],
         [1., 0.],
         [1., 1.]]

    Y = [
        [0.],
        [1.],
        [1.],
        [0.]
    ]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    net = MyNet()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = net(X)
        print("result:{}".format(y_pred))

        loss = loss_fn(y_pred, Y)
        loss.backward()
        print("loss:{:.3f}".format(loss))
        optimizer.step()



