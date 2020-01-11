# coding:UTF-8
import torch

"""
GPUの情報を取得するサンプルプログラム
"""
# GPUの数を取得
gpu_num = torch.cuda.device_count()
print("GPUの数:{}".format(gpu_num))

# GPUの名前を取得
for i in range(gpu_num):
    name = torch.cuda.get_device_name(i)
    print(name)

# カレントのデバイス番号
index = torch.cuda.current_device()
print("current device {}".format(index))

# CUDAを使用できるか
if torch.cuda.is_available():
    print("CUDAを使用できます")
else:
    print("CUDAを使用できません")



