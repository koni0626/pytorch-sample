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
from PIL import Image
import glob
import os
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import tqdm

class ImageDataSet(data.Dataset):
    """
    ジェネレータークラス
    """
    def __init__(self, root_dir, transform=None, phase='train'):
        file_list = self.load_images(root_dir)
        self.file_list = file_list
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
        img = Image.open(img_path)
        img = img.resize((64, 64))
        img = img.convert('RGB')
        img = self.transform(img)
        
        return img


    def load_images(self, root_dir):

        return glob.glob(os.path.join(root_dir, "*.*"))
