## 找出每层随机特征
## 一种是特征均值，一种是二维特征中选取随机特征

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
