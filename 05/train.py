# coding:UTF-8
import argparse
import os
import glob
import cv2
import numpy as np
import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms

from srcnn_model import srcnn_net
"""
画像を拡大するネットワーク
"""

class SrcnnDataSet(data.Dataset):
    """
    ジェネレータークラス
    """
    def __init__(self, file_list, transforms=None):
        self.img_file_list = file_list
        self.transform = transforms


    def __len__(self):
        return len(self.img_file_list)

    def __getitem__(self, index):
        img = cv2.imread(self.img_file_list[index])
        img = cv2.resize(img, (960, 1280)) #あまりに大きい画像が来るとエラーになるため
        o_h, o_w = img.shape[0:2]
        img = img.astype(np.float32)
        img = np.array(img).T/255.
        
        
        anno_img = cv2.imread(self.img_file_list[index])
        anno_img = cv2.resize(anno_img, (240, 320))
        anno_img = cv2.resize(anno_img, (o_w, o_h))
        anno_img = anno_img.astype(np.float32)
        anno_img = anno_img.T/255.
        
        return img, anno_img
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="srcnn")
    parser.add_argument('--train_dir', help="学習データがあるディレクトリ", default='train')
    parser.add_argument('--val_dir', help="検証用データがあるディレクトリ", default='val')
    parser.add_argument('--weights_dir', help="pthファイルの出力先ディレクトリ", default='weights')
    parser.add_argument('--batch_size', help="ミニバッチのサイズ", default=4)
    parser.add_argument('--device', help="GPU", default="cuda:1")
    parser.add_argument('--epochs', help="GPU", default=30)
    
    args = parser.parse_args()

    train_path = args.train_dir
    val_path = args.val_dir
    weights_path = args.weights_dir
    batch_size = args.batch_size
    device = args.device
    epochs = args.epochs

    # 訓練用画像のファイルリスト取得
    train_img_list = glob.glob(os.path.join(train_path, "*.jpg"))

    # 検証用画像のファイルリスト取得
    val_img_list = glob.glob(os.path.join(val_path, "*.jpg"))

    # ジェネレーター作成
    train_dataset = SrcnnDataSet(train_img_list, transforms=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    net = srcnn_net().to(device)
    net.train()

    # 損失関数は最小二乗法
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(net.parameters())

    for epoch in range(epochs):
        print("epoch:{}/{}".format(epoch, epochs))
        loss_val = 0.0
        for i, (img, anno_img) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            x = img.to(device)
            y = anno_img.to(device)
            p = net(x)
            loss = loss_fn(x, p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()

        print("loss:{:.05f}".format(loss_val))
        output_pth = os.path.join(weights_path, "srcnn-{}.pth".format(epoch))
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        torch.save(net.state_dict(), output_pth)
        
        
         
        
    
