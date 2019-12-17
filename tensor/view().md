# view()

>转换维度,每个参数是1个维度,-1代表该维度满元素 

```py

import torch
a= torch.randn(4,4)#16个元素
a.view(2,8)#转为2行8列

a.view(-1)#转为1行16列,等同于a.view(16),[16]

a.view(1,-1)#转为[1,16]

a.view(-1,1)#转为[16,1]




```