# coding:UTF-8
import numpy as np
import torch
from torch import nn
import cv2

device = "cuda:0"

"""
出力サイズの計算式
OH = (H + 2P -FH)/S + 1
OW = (W + 2P -FW)/S + 1
Pはpadding
FH,FWはフィルタのサイズ
Sはストライド
Hは入力サイズ

padding:1 カーネル:2のとき
OH = (3 + 0 - 2)/1 + 1 = 2
padding:2 カーネルサイズ2の時
OH = (3 + 2 - 2)/1 + 1 = 4
"""

# コンボリューションのテスト

net = nn.Sequential(
        # 左からチャンネル数, 枚数, カーネルのサイズ
        nn.Conv2d(3, 3, 2, padding=0)
    )

X = [[[1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.]],
     [[1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.]],
     [[1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.]]]

# データ作成
X = torch.tensor([X])
print(X.size())

# Conv2dで計算
pred_y = net(X)
print(pred_y)

# conv2dのサイズ
print(pred_y.size())

# conv2dのウェイト確認
print(net[0].weight)


# 画像を読み込む
img = cv2.imread(r"data/lenna.jpg")

# 画像をnumpyに変換
img = np.array(img)

# H, W, Cで取得
print(img.shape)

# 画像を32×32にリサイズ
img = cv2.resize(img, (32, 32))
# 画像サイズを(32, 32, 3)に変換する
# 255で割り、で1以下にし、チャンネルを先頭に移動
img = img.T/255.

# tensorに変換
img = torch.tensor([img], dtype=torch.float32)
print(img.size())
y_pred = net(img)
print(y_pred.size())

y_pred = y_pred.detach().numpy()
out_img = y_pred[0]
out_img = out_img.T * 255
out_img = np.array(out_img, dtype=np.uint8)
cv2.imwrite("test.jpg", out_img)
