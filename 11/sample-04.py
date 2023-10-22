# coding: UTF-8
import numpy as np
import glob
import torch
from torch import nn
from torchvision import transforms
import torch.utils.data as data
from PIL import Image


class UNetDataSet(data.Dataset):
    """
    ジェネレータークラス
    """

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()])

        self.train_images = glob.glob(r"../data/11-sample01-unet/train/*.jpg")
        self.train_images = sorted(self.train_images)
        self.train_mask_images = glob.glob(r"../data/11-sample01-unet/train_mask/*.jpg")
        self.train_mask_images = sorted(self.train_mask_images)

    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.train_images)

    def __getitem__(self, index):
        # index番目の画像をロード
        img = Image.open(self.train_images[index])
        img = img.convert("RGB")  # 3チャンネルに変換
        # 画像の前処理(リサイズなど)
        x_image = self.transform(img)
        # マスク値の取得
        img = Image.open(self.train_mask_images[index])
        np_img = np.array(img)
        np_img = np.where(np_img > 10, 1, 0)
        pil_img = Image.fromarray(np_img)
        y_image = self.transform(pil_img)

        return x_image, y_image


# 畳み込みとバッチ正規化と活性化関数Reluをまとめている
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down_pooling():
    return nn.MaxPool2d(2)


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        # 転置畳み込み
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# UNet https://obgynai.com/unet-semantic-segmentation/
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        # 資料中の『FCN』に当たる部分
        self.conv1 = conv_bn_relu(input_channels, 64)
        self.conv2 = conv_bn_relu(64, 128)
        self.conv3 = conv_bn_relu(128, 256)
        self.conv4 = conv_bn_relu(256, 512)
        self.conv5 = conv_bn_relu(512, 1024)
        self.down_pooling = nn.MaxPool2d(2)

        # 資料中の『Up Sampling』に当たる部分
        self.up_pool6 = up_pooling(1024, 512)
        self.conv6 = conv_bn_relu(1024, 512)
        self.up_pool7 = up_pooling(512, 256)
        self.conv7 = conv_bn_relu(512, 256)
        self.up_pool8 = up_pooling(256, 128)
        self.conv8 = conv_bn_relu(256, 128)
        self.up_pool9 = up_pooling(128, 64)
        self.conv9 = conv_bn_relu(128, 64)
        self.conv10 = nn.Conv2d(64, output_channels, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # 正規化
        # x = x / 255.

        # 資料中の『FCN』に当たる部分
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # 資料中の『Up Sampling』に当たる部分, torch.catによりSkip Connectionをしている
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = torch.sigmoid(output)

        return output


def main():
    epochs = 1
    train_dataset = UNetDataSet()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    net = UNet(3, 1)
    net.to("cuda:0")
    net.train()

    for epoch in range(epochs):
        for x, y in train_dataloader:
            x = x.to("cuda:0")
            pred = net(x)
            print(pred)
            # TODO 損失関数とか


if __name__ == '__main__':
    main()
