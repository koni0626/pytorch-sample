# coding:UTF-8
import argparse
import glob
import os
import cv2
import numpy as np
import os

import torch
from srcnn_model import srcnn_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="srcnn")
    parser.add_argument('--image_dir', help="画像のデータパス", default='images')
    parser.add_argument('--output_dir', help="予測結果出力ディレクトリ", default='output')
    parser.add_argument('--weight_path', help="pthファイル", required=True)
    parser.add_argument('--device', help="GPU", default="cuda:1")

    args = parser.parse_args()

    image_dir = args.image_dir
    output_dir = args.output_dir
    weight_path = args.weight_path
    device = args.device
    
    net = srcnn_net()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("{}を読み込みます".format(weight_path))
    net.load_state_dict(torch.load(weight_path))
    net.to(device)
    net.eval()
    
    img_list = glob.glob(os.path.join(image_dir, "*.*"))

    for img_file_path in img_list:
        img = cv2.imread(img_file_path)
        img = cv2.resize(img, (960, 1280))
        img = img.T / 255.
        img = torch.tensor(img, device=device, dtype=torch.float32)
        img = img.unsqueeze(0)

        outputs = net(img)
        outputs = outputs.to("cpu")
        y = outputs[0]
        y = y[0].detach().numpy()
        y = y.T * 255
        y = y.astype(np.uint8)

        img_filename = img_file_path.split(os.sep)[-1]
        output_file = os.path.join(output_dir, img_filename)
        cv2.imwrite(output_file, y)
        print("{}に保存しました".format(output_file))
        
    
