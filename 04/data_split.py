# coding:UTF-8
import glob
import os
import shutil
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
    img_list, label_list = load_images("font_cls")
    train_dir = "data/train"
    val_dir = "data/val"
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # ディレクトリ作成
    for target_dir in [train_dir, val_dir]:
        for i in range(28):
            dst_dir = os.path.join(target_dir, "{0:04d}".format(i))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
    
    for img_name, label in zip(img_list, label_list):
        sel = random.randint(0, 100)

        dst_dir = train_dir
        if sel > 70:
            dst_dir = val_dir
        
        dst_dir = os.path.join(dst_dir, "{0:04d}".format(label))
        shutil.copy(img_name, dst_dir)
        print("{}を{}にコピーしました".format(img_name, dst_dir))
        
        

        
        print("{},{}".format(img_name, label))
        
