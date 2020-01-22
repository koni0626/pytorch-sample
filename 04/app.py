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

class ImageTransform():
    """
    画像変換クラス
    """
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)]),
            'val': transforms.Compose([
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
        }
    
    
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class NumberDataSet(data.Dataset):
    """
    ジェネレータークラス
    """
    def __init__(self, file_list, labels, transform=None, phase='train'):
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
        img = Image.open(img_path)
        img = img.convert("RGB") #3チャンネルに変換
        #画像の前処理(リサイズなど)
        img_transformed = self.transform(img, self.phase)
        
        #ラベルの取得
        label = self.labels[index]
        #ラベルってcategoricalにしなくてよいの？
        
        return img_transformed, label


def load_images(root_dir):
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


def train_model(net, data_loaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0")
    net.to(device)
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch+1, num_epochs))
        print('------------')
        for phase in ['train', 'val']:
           # print(phase)
            if phase == 'train':
                net.train() #訓練モード
            else:
                net.eval()
        
            epoch_loss = 0.0
            epoch_corrects = 0
            
            #未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch %10 != 0) and (phase == 'val'):
                continue
            
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                #optimizerを初期化
                optimizer.zero_grad()
                
                # 順伝搬の計算
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = net(inputs)
                    loss = criterion(outputs, labels) #損失計算
                    _, preds = torch.max(outputs, 1) #ラベルを予測
                  #  print(preds)
                   # print(phase)
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()
                    
                    #イテレーション結果の計算
                    #lossの合計を計算
                    epoch_loss += loss.item() * inputs.size(0) #32
                    #正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss:{:.4f} Acc:{:4f}'.format(phase, epoch_loss, epoch_acc))
        weight_file_name = "weights/{}.weights".format(epoch)
        torch.save(net.state_dict(),weight_file_name)
        
        
if __name__ == '__main__':
    resize = (224, 224)
    mean = (0.55, 0.554, 0.55)
    std = (0.253, 0.252, 0.25)
    train_img_list, train_label_list = load_images('data/train')
    val_img_list, val_label_list = load_images('data/val')
    
    train_dataset = NumberDataSet(file_list=train_img_list, labels=train_label_list, transform=ImageTransform(resize, mean, std), phase='train')
    val_dataset = NumberDataSet(file_list=val_img_list, labels=val_label_list, transform=ImageTransform(resize, mean, std), phase='val')
    
    batch_size=32
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}
    inputs, labels = next(iter(train_dataloader))
    
    
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=28)
    
    #訓練モード
    net.train()

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()
    
    # 学習させるパラメーター名
    update_param_names = ["classifier.6.weight", "classifier.6.bias"]
    params_to_update = []
    
    #学習させるパラメーター以外は勾配計算をなくし、変化しないように設定
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
            print(name)
        else:
            param.requires_grad = True
    print(params_to_update)
    
    # 最適化手法の設定
    #optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(params_to_update, lr=learning_rate)

    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=200)
    
