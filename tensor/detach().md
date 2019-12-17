# tensor的detach()函数和data属性

当一个tensor的requires_grad=True时，它的复制变量detach()和data一个不可导，一个可导

```py
t = torch.tensor([0., 1.], requires_grad=True)
t2 = t.detach()
t3 = t.data
print(t2.requires_grad, t3.requires_grad)  # ouptut: False, False
```

不可导
```py
>>> a = torch.tensor([1,2,3.], requires_grad = True)
>>> out = a.sigmoid()
>>> c = out.detach()
>>> c.zero_()  
tensor([ 0.,  0.,  0.])

>>> out  # modified by c.zero_() !!
tensor([ 0.,  0.,  0.])

>>> out.sum().backward()  # Requires the original value of out, but that was overwritten by c.zero_()
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

可导，但是导数错误
```py
>>> a = torch.tensor([1,2,3.], requires_grad = True)
>>> out = a.sigmoid()
>>> c = out.data
>>> c.zero_()
tensor([ 0.,  0.,  0.])

>>> out  # out  was modified by c.zero_()
tensor([ 0.,  0.,  0.])

>>> out.sum().backward()
>>> a.grad  # The result is very, very wrong because `out` changed!
tensor([ 0.,  0.,  0.])
```

- 总结
tensor的变量a加detach(即a.detach())是防止变量变量改变的时候，计算出错误的梯度.


https://github.com/pytorch/pytorch/issues/6990
https://zhuanlan.zhihu.com/p/83329768
