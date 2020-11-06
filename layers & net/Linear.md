## 1.nn.Linear() 偏函数

格式：Linear(in_dim,out_dim)

类似矩阵和向量相乘,得到另一个向量.

当计算图片时等二维数据时，需要将其拉成一维, 这样才使用于向量计算，如[n,m] -> [n*m]

>[n,in_dim]*[in_dim,out_dim]=[out_dim,n]

全联接在图像操作中是按像素操作的，一个结点是一个像素，n*n个像素到m*m个像素的全连接计算就需要有n*n*m*m个参数，分辨率大的时候计算量相当大。
而卷积计算参数只是卷积核的结点数，这也就是为什么高分辨率图像生成任务要避免过大的卷积核，多用3*3的卷积核运算，再增加深度，就是为了减少计算量。


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

## 2.F.near(x,weight,bias) 函数

>import torch.nn.functional as F

> y = F.linear(x,weight,bias)

> y->[-1,m] ,  x->[-1,n]  , weight->[n,m] , bias -> [-1,m]



## Reference

https://pytorch.org/docs/master/nn.html#linear-layers

https://blog.csdn.net/m0_37586991/article/details/87861418
