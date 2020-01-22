# coding:UTF-8
import numpy as np
import torch
import torchvision
from torchvision import models, transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import cv2
import glob
import os
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import tqdm

class NumberDataSet(data.Dataset):
    """
    ジェネレータークラス
    """
    def __init__(self, root_dir, transform=None, phase='train'):
        file_list, labels = self.load_images(root_dir)
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        self.phase = phase
    
    
    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)
    
    
    def __getitem__(self, index):
        #index番目の画像をロード
        img_path = self.file_list[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.T / 255.
        #float64からfloat32に変換
        img = img.astype(np.float32)
        #ラベルの取得
        #print(img.dtype)
        label = self.labels[index]
        #ラベルってcategoricalにしなくてよいの？
        
        return img, label


    def load_images(self, root_dir):
        dir_list = glob.glob(os.path.join(root_dir, "*"))
        dir_list.sort()
        img_list = []
        label_list = []
        for i, dirname in enumerate(dir_list):
            file_list = glob.glob(os.path.join(dirname, "*.*"))
            
            for j, img_name in enumerate(file_list):
                
                label_list.append(i)
                img_list.append(img_name)
            
        return img_list, label_list



class FlattenLayer(nn.Module):
    def forward(self, x):
        sizes = x.size()
        #print(sizes)
        return x.view(sizes[0], -1)


def create_net(classes):
    conv_net = nn.Sequential(
        nn.Conv2d(3, 32, 5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.25),
        nn.Conv2d(32, 64, 5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.25),
        FlattenLayer()
    )
    
    # あとで出力を調べる
    test_input = torch.ones(1,3,32,32)
    conv_output_size = conv_net(test_input).size()[-1]
    print(conv_output_size)
    mlp = nn.Sequential(
        nn.Linear(conv_output_size, 200),
        nn.ReLU(),
        nn.BatchNorm1d(200),
        nn.Dropout(0.25),
        nn.Linear(200, classes)
    )

    net = nn.Sequential(
        conv_net,
        mlp
    )
    
    return net


def eval_net(net, data_loader, device="cpu"):
    # DropoutやBatchNormを無効化
    net.eval()
    
    ys = []
    ypreds = []
    
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        
        # 自動微分をOFFにする
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
    
    # ミニバッチの予測結果を一つにまとめる    
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    
    acc = (ys == ypreds).float().sum()/ len(ys)
    return acc.item()


def train_net(net, train_loader, test_loader, optimizer_cls = optim.Adam, loss_fn = nn.CrossEntropyLoss(), n_iter=10, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        net.train()
        n = 0
        n_acc = 0
        
        for i,(xx,yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        #for i,(xx,yy) in enumerate(train_loader):
            xx = xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
            _, y_pred = h.max(1)

            n_acc += (yy==y_pred).float().sum().item()
        train_losses.append(running_loss/i)
        
        train_acc.append(n_acc / n)
        val_acc.append(eval_net(net, test_loader, device))
        print("epoch:{} train_loss:{:.3f} train_acc:{:.3f} val_acc{:.3f}".format(epoch, train_losses[-1], train_acc[-1], val_acc[-1]), flush=True)
        weight_file_name = "weights/{}.weights".format(epoch)
        torch.save(net.state_dict(),weight_file_name)


if __name__ == '__main__':
    batch_size = 32
    device = "cuda:0"
    
    train_dataset = NumberDataSet(root_dir='data/train', transform=transforms.ToTensor(), phase='train')
    test_dataset = NumberDataSet(root_dir='data/val', transform=transforms.ToTensor(), phase='val')
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    net = create_net(28)
    load_weights = torch.load("weights/188.weights")
    net.load_state_dict(load_weights)
    net = net.to(device)
    net.eval()
    
    img = cv2.imread("data/val/0010/3de19ea3-bbc8-4a96-975c-3b18bd735cb4.png")
    img = cv2.resize(img, (32, 32))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.T / 255.

    #float64からfloat32に変換
    img = img.astype(np.float32)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze_(0)
    img = img.to(device)
    h = net(img)
    a, y_pred = h.max(1)
    print(a)
    print(y_pred)
    #summary(net, input_size=(3, 32, 32))

    
    
    
    
