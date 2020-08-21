# 浅谈torch.nn库和torch.nn.functional库
这两个库很类似，都涵盖了神经网络的各层操作，只是用法有点不同，

## 1.nn

nn是类实现，即在nn下计算层是一个类，一般继承nn.module通过定义forward()函数计算其值

- conv1d
```
class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return torch.nn.functional.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
```
## 2.nn.functional

nn.functional下是通过函数实现计算层，将层中的可训练参数传入函数functional的参数即可使用，其会直接返回一个torch.nn.functional的函数，和上面类中的forward()中的函数一致

- conv1d

```py
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    if input is not None and input.dim() != 3:
        raise ValueError("Expected 3D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = ConvNd(_single(stride), _single(padding), _single(dilation), False,
               _single(0), groups, torch.backends.cudnn.benchmark,
               torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)
```

- 注意
1.nn.Xxx不需要自己定义和管理参数weight；而nn.functional.xxx需要自己定义weight，每次调用的时候都需要手动传入weight
2.另外droptout用nn定义的话在训练时生效，在eval()时无效。

## 3.示例

损失函数Loss(交叉熵)

```py
# nn库
import torch
import torch.nn as nn

Loss = nn.BCELoss()

a = torch.ones(2,2)
b = torch.ones(2,2)
c = Loss(a,b)

# nn.functional库
import torch
import torch.nn.functional as nn

a = torch.ones(2,2)
b = torch.ones(2,2)
c = nn.binary_cross_entropy(a,b)
```

c的结果都一样为0，即两个分布高度相似

总结一下，两个库都可以实现神经网络的各层运算。其他包括卷积、池化、padding、激活(非线性层)、线性层、正则化层、其他损失函数Loss，两者都可以实现

nn.functional.xxx是函数接口，而nn.Xxx是nn.functional.xxx的类封装，并且nn.Xxx都继承于一个共同祖先nn.Module。

因此nn.Xxx除了具有nn.functional.xxx功能(通过类中的forward方法实现)，内部附带了nn.Module相关的属性和方法，例如train(), eval(),load_state_dict, state_dict 等,可以自动管理各层的参数,同时还可以实现如Sequential()将多个运算层组合为一个逻辑层。

参考
https://pytorch.org/docs/stable/nn.html#

https://pytorch.org/docs/stable/nn.functional.html#

https://www.zhihu.com/question/66782101
