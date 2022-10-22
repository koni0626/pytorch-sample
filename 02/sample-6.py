# coding: UTF-8
import csv
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


def create_data():
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

    return x, y


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.fc1_weight_memory = []
        self.f = open("weight.csv", "w")

    def __del__(self):
        self.f.close()

    def forward(self, x):

        sel_num = 8
        input_size = 2
        record = ""
        for s in range(sel_num):
            for i in range(input_size):
                record += str(self.fc1.weight[s][i].item()) + ","
            record += str(self.fc1.bias[s].item()) + ","
        self.f.write(record[0:-1]+"\n")
        print(record[0:-1])

        x = self.fc1(x)
        # 4bat文まとめて計算しているから4つ出ている。
        x = self.relu1(x)
        #print(x)
        x = self.fc2(x)
        #print(x)
        x = self.sigmoid(x)
        #print(x)
        return x

    def disp_weight(self):
        print(self.fc1.weight)
        print(self.fc1.bias)


def disp_graph():
    fig = plt.figure()
    anime_list = []
    records = []
    with open("weight.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            records.append(row)

    for record in records:
        x = [i for i in range(16+8)]
        record = [float(d) for d in record]
        im = plt.bar(x, record, color="blue")
        #plt.show()
        anime_list.append(im)

    ani = ArtistAnimation(fig, anime_list, interval=100)
    ani.save('animation.gif', writer='pillow')


def main():
    x, y = create_data()
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    net = MyNet()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        # 予測
        pred = net(x)

        # loss計算
        loss = loss_fn(pred, y)
        # acc計算
        cls = torch.where(pred >= 0.5, torch.tensor(1.), torch.tensor(0.))
        correct_sum = torch.sum(cls == y, dtype=torch.float32)
        acc = correct_sum / y.size()[0]

        print(f"loss:{loss:.3f} acc:{acc:.3f}")

        loss.backward()
        optimizer.step()

        print(pred)


if __name__ == '__main__':
    main()
    disp_graph()