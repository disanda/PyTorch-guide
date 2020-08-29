# 不算梯度(no_grad)

当不训练,即不更新模型参数时，如推断，验证时。不需要反向传播(BP)，也就不需要求梯度。

为了不求梯度(这个计算量很大),有以下两种方式

## torch.no_grad()
```py
import torch 
conv = torch.nn.Conv2d(1,3,1)
x = torch.randn(1,1,8,8)
with torch.no_grad():
  y = conv(x)
flag = y.reguires_grad
print(flag) # False
```

## detach()

通过tensor.detach()赋值给新变量，让新变量和原变量涉及的模型梯度计算解耦

```py
conv = torch.nn.Conv2d(1,3,1)
x = torch.randn(1,1,8,8)
y = conv(x)
z = y.detach()
flag = z.requires_grad # Flase
print(flag)
```
