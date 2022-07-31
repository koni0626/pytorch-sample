# coding UTF-8
from collections import OrderedDict

dict = {}
dict['z'] = 1
dict['y'] = 2
dict['x'] = 3
for key, val in dict.items():
    print(key, val)

dict = OrderedDict()
dict['z'] = 1
dict['y'] = 2
dict['x'] = 3

for key, val in dict.items():
    print(key, val)

# 変わらないな…