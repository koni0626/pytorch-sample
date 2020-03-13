# coding:UTF-8
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader
from ImageDataSet import ImageDataSet
from PIL import Image
import numpy as np


IMG_SIZE = 64
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(IMG_SIZE*IMG_SIZE*3, 4096)
        self.fc21 = nn.Linear(4096, 20)
        self.fc22 = nn.Linear(4096, 20)
        self.fc3 = nn.Linear(20, 4096)
        self.fc31 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, IMG_SIZE*IMG_SIZE*3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h31 = F.relu(self.fc31(h3))
        return torch.sigmoid(self.fc4(h31))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, IMG_SIZE*IMG_SIZE*3))
        z = self.reparameterize(mu, logvar)
        return z#, self.decode(z), mu, logvar

def src_img(img_path, device):
    img = Image.open(img_path)
    img = img.resize((64, 64))
    img = img.convert('RGB')
    #img = np.asarray(img)
    t = transforms.ToTensor()
    img = t(img).to(device)
    return img #torch.tensor(img, device=device, dtype=torch.float32)
    
    
if __name__ == '__main__':
    model = VAE()
    device = "cuda:0"
    model.load_state_dict(torch.load("weights/990.pth"))
    model.to(device)
    with torch.no_grad():
        img1 = src_img("/home/konishi/data/yurudora/1345010101.png", device)
        img1 = img1.view(1, 3, IMG_SIZE, IMG_SIZE)
        z1 = model(img1)
        
        img2 = src_img("/home/konishi/data/yurudora/1248010101.png", device)
        img2 = img2.view(1, 3, IMG_SIZE, IMG_SIZE)
        z2 = model(img2)
        
        for i in range(100):
            z3 = (i*z2 + (100-i)*z1) / 100
            sample = model.decode(z3).cpu()
            save_image(sample.view(1, 3, IMG_SIZE, IMG_SIZE), 'play/play{}.jpg'.format(i))
        #sample = torch.randn(64, 20).to(device)
        #sample = model.decode(z).cpu()
        #save_image(recon_batch.view(1, 3, IMG_SIZE, IMG_SIZE), 'play/play.jpg')

