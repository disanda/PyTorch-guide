# PyTorch教学库


https://pytorch.org/docs/stable/optim.html


## 浅谈torch.nn库和torch.nn.functional库

这两个库很类似，都涵盖了神经网络的各层操作，只是用法有点不同，比如在损失函数Loss中实现交叉熵

- nn库
```py

import torch
import torch.nn as nn

Loss = nn.BCELoss()

a = torch.ones(2,2)
b = torch.ones(2,2)
c = Loss(a,b)

```

- nn.functional库

```py

import torch
import torch.nn.functional as nn

a = torch.ones(2,2)
b = torch.ones(2,2)
c = nn.binary_cross_entropy(a,b)

```

>c的结果都一样为0，即两个分布高度相似

总结一下，两个库都可以实现神经网络的各层运算。其他包括卷积、池化、padding、激活(非线性层)、线性层、正则化层、其他损失函数Loss，两者都可以实现

不过nn.functional毕竟只是nn的子库，nn的功能要多一些，还可以实现如Sequential()这种将多个层弄到一个序列这样牛逼的骚操作。
