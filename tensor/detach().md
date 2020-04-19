# detach
detach是一个对tensor变量解藕的函数，当变量用到了detach, 那么它就不在可导，其在计算图中的model也就无法更新梯度(用于减少计算量)。
从而也就不能更新对应model中的参数。

可用于训练时只更新部分网络梯度，如下A,B网络
```python
# y=A(x), z=B(y) 求B中参数的梯度，不求A中参数的梯度
y = A(x)
z = B(y.detach())
z.backward()
```
用了detach(),在计算z.backward()时，就不会再去求和y有关的网络A的梯度，只求B()的梯度，也就是只更新B(),从而减轻计算量



## tensor的detach()函数和data属性

当一个tensor的requires_grad=True时，它的复制变量detach()和data一个不可导，一个可导

```py
t = torch.tensor([0., 1.], requires_grad=True)
t2 = t.detach()
t3 = t.data
print(t2.requires_grad, t3.requires_grad)  # ouptut: False, False
```


- 参考
https://www.cnblogs.com/jiangkejie/p/9981707.html
https://github.com/pytorch/pytorch/issues/6990
https://zhuanlan.zhihu.com/p/83329768
