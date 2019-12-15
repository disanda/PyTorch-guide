## nn.Linear()函数

> y = xA^t+b

```py
import torch

x = torch.randn(128, 20)  # 输入的维度是（128，20）
m = torch.nn.Linear(20, 30)  # 20,30是指参数矩阵的转置维度
output = m(x) # 维度为 128,30
print('m.weight.shape:\n ', m.weight.shape) #参数矩阵的维度:(30,20)
print('m.bias.shape:\n', m.bias.shape) #(30)

# ans = torch.mm(input,torch.t(m.weight))+m.bias 等价于下面的
ans = torch.mm(x, m.weight.t()) + m.bias   
print('ans.shape:\n', ans.shape)

print(torch.equal(ans, output))
```


## Reference

https://pytorch.org/docs/master/nn.html#linear-layers

https://blog.csdn.net/m0_37586991/article/details/87861418
