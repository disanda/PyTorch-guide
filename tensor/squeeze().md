# squeeze()

去除张量中元素个数为1的维度,参数为具体维度下标(0开始)

```py

import torch

a=torch.randn(4,4) #[4,4]

a1 = a.view(1,-1) # a1.shape=[1,16]
a2 = a.view(-1,1) # a2.shape=[16,1]


a1 = a1.squeeze(0) #a1.shape=[16]
a2 = a2.squeenze(1) #a2.shape=[16]
#去除了为1的维度

```



