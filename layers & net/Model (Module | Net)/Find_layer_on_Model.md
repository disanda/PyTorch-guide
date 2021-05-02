## 找出每层随机特征

> 如下是示例，关键是处理字典

```py
import torch
import random

#针对网络G
def findF_G(model,x):
    y = []
    y.append(x)
    for name, layer in model.net._modules.items():
        x = layer(x)
        if isinstance(layer, torch.nn.ConvTranspose2d):
            #print(x.shape)
            y.append(x.mean([2,3]))
    y[0]=y[0].squeeze(2).squeeze(2)
    y[-1]= x
    return y
```
