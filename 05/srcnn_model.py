# coding:UTF-8
import torch.nn as nn

def srcnn_net():
    srcnn_net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4),
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
        nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2),
        nn.ReLU()
    )
    
    
    return srcnn_net
