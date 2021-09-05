# coding:UTF-8
import random
import numpy as np
import glob

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms
from torchvision.utils import save_image


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        #Encoder Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=512,
                               kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256,
                               kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128,
                               kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64,
                               kernel_size=3, padding=1)
        #Decoder Layers
        self.t_conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=128,
                                          kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=256,
                                          kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=512,
                                          kernel_size=2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(in_channels=512, out_channels=3,
                                          kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #コメントに28×28のモノクロ画像をi枚を入力した時の次元を示す
        #encode#                          #in  [i, 3, 64, 64]
        x = self.relu(self.conv1(x))      #out [i, 512, 64, 64]
        x = self.pool(x)                  #out [i, 512, 32, 32]
        print(x.shape)
        x = self.relu(self.conv2(x))      #out [i, 512, 32, 32]
        x = self.pool(x)                  #out [i ,256, 16, 16]
        x = self.relu(self.conv3(x))      #out [i, 128, 16, 16]
        x = self.pool(x)                  #out [i ,128, 8, 8]
       # print(x.shape)
        x = self.relu(self.conv4(x))      #out [i, 64, 4, 4]
        x = self.pool(x)                  #out [i ,64, 4, 4]
       # print(x.shape)
        #decode#
        x = self.relu(self.t_conv1(x))    #out [i, 128, 8, 8]
        x = self.relu(self.t_conv2(x))    #out [i, 256, 16, 16]
        x = self.relu(self.t_conv3(x))    #out [i, 512, 32, 32]
        x = self.sigmoid(self.t_conv4(x)) #out [i, 3, 64, 64]
        return x


def train_net(n_epochs, train_loader, net, optimizer_cls = optim.Adam, loss_fn = nn.MSELoss(), device=torch.device("cuda:0")):
    """
    n_epochs…訓練の実施回数
    net …ネットワーク
    device …　"cpu" or "cuda:0"
    """
    losses = []         #loss_functionの遷移を記録
    optimizer = optimizer_cls(net.parameters(), lr = 1e-4)
    net.to(device)
    for epoch in range(n_epochs):
        running_loss = 0.0
        net.train()

        for i, XX in enumerate(train_loader):
            XX = XX.to(device)
            optimizer.zero_grad()

            XX_pred = net(XX)             #ネットワークで予測

            try:
                save_image(XX_pred, r"E:\tmp1\{:03d}.jpg".format(epoch))
            except:
                pass

            loss = loss_fn(XX, XX_pred)   #予測データと元のデータの予測
            loss.backward()
            optimizer.step()              #勾配の更新
            running_loss += loss.item()

        losses.append(running_loss / i)
        torch.save(net.state_dict(), r"weights\anime_{}.pth".format(epoch))
        print("epoch", epoch, ": ", running_loss / i)

    return losses


class CAEDataSet(data.Dataset):
    """
    ジェネレータークラス
    """

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)

    def __getitem__(self, index):
        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)
       # img = img.convert("RGB")  # 3チャンネルに変換

        img_transformed = self.transform(img)

        return img_transformed


if __name__ == '__main__':
    device = torch.device("cuda:0")
    resize = (64, 64)
    batch_size = 32
    train_img_list = glob.glob(r"E:\anime\*.png")

    train_dataset = CAEDataSet(file_list=train_img_list, transform=transforms.Compose([
                                transforms.Resize(resize),
                                transforms.ToTensor()]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ConvAutoencoder()
    model.train()

    losses = train_net(n_epochs=1000,
                       train_loader=train_dataloader,
                       net=model, device=device)
