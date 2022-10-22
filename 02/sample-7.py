# coding: UTF-8
import numpy as np
import torch
import torchvision.transforms.functional
from torch import nn
from torch import optim
from torchvision import models, transforms
from PIL import Image

index = 0

def print_model(module, name="model", depth=0):
    if len(list(module.named_children())) == 0:
        print(f"{' ' * depth} {name}: {module}")
    else:
        print(f"{' ' * depth} {name}: {type(module)}")

    for child_name, child_module in module.named_children():
        if isinstance(module, torch.nn.Sequential):
            child_name = f"{name}[{child_name}]"
        else:
            child_name = f"{name}.{child_name}"
        print_model(child_module, child_name, depth + 1)


def forward_hook(module, inputs, outputs):
    # 順伝搬の出力を features というグローバル変数に記録する
    global features
    global index
    # 1. detach でグラフから切り離す。
    # 2. clone() でテンソルを複製する。モデルのレイヤーで ReLU(inplace=True) のように
    #    inplace で行う層があると、値がその後のレイヤーで書き換えられてまい、
    #    指定した層の出力が取得できない可能性があるため、clone() が必要。
    features = outputs.detach().clone()
    img = feature_to_img(features[0][:16])
    img.save(f"result/{index}.jpg")
    index += 1


def feature_to_img(feature, nrow=4):
    # (N, H, W) -> (N, C, H, W)
    feature = feature.unsqueeze(1)
    # 画像化して、格子状に並べる
    img = torchvision.utils.make_grid(feature.cpu(), nrow=nrow, normalize=True, pad_value=1)
    # テンソル -> PIL Image
    img = transforms.functional.to_pil_image(img)
    # リサイズする。
    new_w = 500
    new_h = int(new_w * img.height / img.width)
    img = img.resize((new_w, new_h))

    return img


def main():
    global index
    model = models.vgg16(pretrained=True)
    print_model(model)
    # 抽出対象の層
    target_module = model.features[index]  # (3): ReLU(inplace=True)
    target_module.register_forward_hook(forward_hook)

    for i in range(31):
        model.features[i].register_forward_hook(forward_hook)

    model.eval()
    img = Image.open("./data/j.jpg")
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
    print(img.shape)
    result = model(img)
    #print(torch.argmax(result_dog))


if __name__ == '__main__':
    main()
