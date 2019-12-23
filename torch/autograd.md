
## torch.autograd

### Variable

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

### grad

计算梯度的函数,可求二阶导
>torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)

- grad_outputs:是参数值
- create_graph:bool类型，确认是否创建梯度图节点，用于算高阶梯度

```py


import torch
from torch.autograd import Variable

x = Variable(torch.randn(3,4),requires_grad=True) 
#x = torch.randn(3, 4).requires_grad_(True)

for i in range(3):
    for j in range(4):
        x[i][j] = i + j
y = x ** 2
print(x)
print(y)
weight = torch.ones(y.size())
print(weight)
dydx = torch.autograd.grad(outputs=y,
                           inputs=x,
                           grad_outputs=weight,
                           retain_graph=True,
                           create_graph=True,
                           only_inputs=True)

# (x**2)' = 2*x 
# dydx的形式是元组

print(dydx[0])
d2ydx2 = torch.autograd.grad(outputs=dydx[0],
                             inputs=x,
                             grad_outputs=weight,
                             retain_graph=True,
                             create_graph=True,
                             only_inputs=True)

```




https://pytorch.org/docs/stable/autograd.html

https://blog.csdn.net/qq_36556893/article/details/91982925

https://www.cnblogs.com/hellcat/p/8453615.html
