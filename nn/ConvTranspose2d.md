# 卷积与反卷积

卷积和反卷积是图片计算在深度学习中常用的上采样和下采样操作。相比其他采样操作，卷积计算不仅可以保存参数的梯度传递(适用用BP),还可以改变图片的通道以更好的整合局部特征。

在torchn.nn中,卷积操作是一个函数，输入为一组图片或特征变量[n,c,w,h],输出也为一组变量[n,c,w,h].变量类型为tensor.

## 1.Conv2d

卷积可以压缩整合图片特征，让通道/宽/高分别为:[c,w,h]的特征图片通过Conv2d。变为更多的通道(维度)c，更小的尺寸W/H.

这里有几个参数比较重要:

- padding

就是填充的意思，通过padding，可以填充图片的边缘，让图片的边缘的特征得到更充分的计算(不至于被截断)

- kernel_size

卷积核尺寸，尺寸越大‘感受野’越大，及处理的特征单位越大，同时计算量也越大

- stide

卷积核移动的步数，默认1步，增大步数会忽略局部细节计算，适用于高分辨率的计算提升

![image.png](![no_padding_no_strides.gif](https://i.loli.net/2020/04/26/hikOAHsaL3mv5jf.gif))

```
import torch
import torch.nn as nn

x = torch.randn(1,1,4,4)
l = nn.Conv2d(1,1,3)#Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),padding=0)
y = l(x) # y.shape:[1,1,2,2]
```

## 2.ConvTranspose2d

转置卷积，也称为反卷积(deconvlution)和分部卷积(fractionally-strided convolution)。为卷积的逆操作，即把特征的维度压缩，但尺寸放大。

>torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

```py
>>> # With square kernels and equal stride
>>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> input = torch.randn(20, 16, 50, 100)
>>> output = m(input)
>>> # exact output size can be also specified as an argument
>>> input = torch.randn(1, 16, 12, 12)
>>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
>>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
>>> h = downsample(input)
>>> h.size()
torch.Size([1, 16, 6, 6])
>>> output = upsample(h, output_size=input.size())
>>> output.size()
torch.Size([1, 16, 12, 12])
```


