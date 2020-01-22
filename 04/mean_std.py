# coding:UTF-8
import glob
import cv2
import os
import numpy as np
import random

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

if __name__ == '__main__':
    img_list, _ = load_images("font_cls")
    
    b = np.array([])
    g = np.array([])
    r = np.array([])
    size = len(img_list)
    for i in range(1000):
        print(i)
        sel = random.randint(0, size-1)
        img = cv2.imread(img_list[sel])
        h, w = img.shape[0:2]
        total = h*w
        # Bだけ抜く
        img = img/255.
        b_img = img[:,:,0].flatten()
        g_img = img[:,:,1].flatten()
        r_img = img[:,:,2].flatten()
        
        b = np.hstack((b, b_img))
        g = np.hstack((g, g_img))
        r = np.hstack((r, r_img))
    

    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)   
    b_std = np.std(b)
    g_std = np.std(g)
    r_std = np.std(r)

    print(b_mean, g_mean, r_mean)
    print(b_std, g_std, r_std)
