
## torch.autograd

- Variable

其中autograd.Variable为核心类，其在tensor的基础上增加了求导(梯度)功能，其由一个tensor(data)，以及tensor的梯度(grad)，
以及tensor的函数关系(grad_fn)构成。

梯度就是自变量的求导值

```py
import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)

y = x + 2

z = y*y*3

o = z.mean()
#求导的因变量必须时一个值而不能是多维数据

o.backward()

x.grad()
#即x=1时的梯度值


```

