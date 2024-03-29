import torch.nn as nn
from torchvision import models, transforms
from collections import OrderedDict


class Example(nn.Module):
    def __init__(self, tasks):
        super(Example, self).__init__()
        self.tasks = tasks

        feedfnn = []
        for task_name, num_class in self.tasks:
            ffnn = nn.Sequential(OrderedDict([
                ('dense1', nn.Linear(10, 10)),
                ('tanh', nn.Tanh()),
                ('dense2', nn.Linear(10, 10)),
                ('tanh', nn.Tanh()),
                ('dense3', nn.Linear(10, 10))
            ]))
            feedfnn.append((task_name, ffnn))
            
        self.ffnn = nn.Sequential(OrderedDict(feedfnn))


if __name__ == '__main__':
    # 自分で名前をつける版
    tasks = (('task1', None), ('task2', None), ('task3', None))
    example = Example(tasks)
    print(example)

    # 比較のためにVGGモデルも出力する
    vgg = models.vgg16(pretrained=True)
    print(vgg)

