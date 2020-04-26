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

### 1.2 卷积操作及可视化 

蓝色为输入，蓝色上的阴影为卷积核(kernel)，绿色为输出，蓝色边缘的白色框为padding

- padding=0,stride=1,kernel_size=3

![image](https://i.loli.net/2020/04/26/hikOAHsaL3mv5jf.gif)

>尺寸从[4,4]->[2,2]

```python
import torch
import torch.nn as nn

x = torch.randn(1,1,4,4)
l = nn.Conv2d(1,1,3)#Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),padding=0)
y = l(x) # y.shape:[1,1,2,2]
```
- padding=2,stride=1,kernel_size=4

![image](https://i.loli.net/2020/04/26/kMnaiNpbAqldIhX.gif)

>尺寸从[5,5]->[6,6]

```python
import torch
import torch.nn as nn

x = torch.randn(1,1,5,5)
l = nn.Conv2d(1,1,4,padding=2)#Conv2d(1, 1, kernel_size=4,stride=1,padding=2)
y = l(x) # y.shape:[1,1,6,6]
```


## 2.ConvTranspose2d

转置卷积，也称为反卷积(deconvlution)和分部卷积(fractionally-strided convolution)。为卷积的逆操作，即把特征的维度压缩，但尺寸放大。

函数形式如下：

>torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

### 2.1 操作及可视化

这里需要注意的是padding和stride和conv2d不同，padding不是蓝色的留白,是kernel像图像中心移动的单位。如下当padding=0时，卷积核刚好和输入边缘相交一个单位。因此pandding可以理解为卷积核向中心移动的步数。 同时stride也不再是kernel移动的步数，变为输入单元彼此散开的步数。

- padding=0,kernel_size=3,stride=1

![J66KbV.gif](https://s1.ax1x.com/2020/04/26/J66KbV.gif)

```python
import torch
import torch.nn as nn

x = torch.randn(1,1,2,2)
l = nn.ConvTranspose2d(1,1,3)#Conv2d(1, 1, kernel_size=3,stride=1,padding=0)
y = l(x) # y.shape:[1,1,4,4]
```

- padding=2,kernel_size=4,stride=1

![J66UDx.gif](https://s1.ax1x.com/2020/04/26/J66UDx.gif)

```python
import torch
import torch.nn as nn

x = torch.randn(1,1,6,6)
l = nn.ConvTranspose2d(1,1,4,padding=2)#Conv2d(1, 1, kernel_size=4,stride=1,padding=2)
y = l(x) # y.shape:[1,1,5,5]
```

- padding=2,kernel_size=3,stride=1

注意这个kernel也是向中心内移了2（对比padding=0），所以padding为2

![J662rt.gif](https://s1.ax1x.com/2020/04/26/J662rt.gif)

```python
import torch
import torch.nn as nn

x = torch.randn(1,1,7,7)
l = nn.ConvTranspose2d(1,1,3,padding=2)#Conv2d(1, 1, kernel_size=3,stride=1,padding=2)
y = l(x) # y.shape:[1,1,5,5]
```
- padding=0,kernel_size=3,stride=2

![J6ceiD.gif](https://s1.ax1x.com/2020/04/26/J6ceiD.gif)

```python
import torch
import torch.nn as nn

x = torch.randn(1,1,2,2)
l = nn.ConvTranspose2d(1,1,3,stride=2,padding=0)#Conv2d(1, 1, kernel_size=3,stride=2,padding=0)
y = l(x) # y.shape:[1,1,5,5]
```

- padding=1,kernel_size=3,stride=2

![J6c8df.gif](https://s1.ax1x.com/2020/04/26/J6c8df.gif)

```python
import torch
import torch.nn as nn

x = torch.randn(1,1,3,3)
l = nn.ConvTranspose2d(1,1,3,stride=2,padding=1)#Conv2d(1, 1, kernel_size=3,stride=2,padding=1)
y = l(x) # y.shape:[1,1,5,5]
```

### 参考

https://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose2d

