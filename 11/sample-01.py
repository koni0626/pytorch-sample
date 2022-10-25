# coding: UTF-8
import torch
from torch import nn
from torchvision import transforms
from PIL import Image


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)

        x2 = self.max_pool(x1)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        x2 = self.up_sample(x2)

        x3 = torch.cat([x1, x2], dim=1)

        x3 = self.conv5(x3)
        x3 = self.conv_out(x3)
        x3 = self.sigmoid(x3)

        return x3


def main():
    net = UNet()
    net.eval()
    img = Image.open("../data/dog.jpg")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = preprocess(img)
    img = torch.unsqueeze(img, 0)

    result = net(img)
    print(result)


if __name__ == '__main__':
    main()
