## 优化器
通过torch.optim下面对各种类(如SGD,Adam)创建优化器对象，该对象能够保持当前参数状态并基于计算得到的梯度更新参数，更新参数分为两步骤.

- 第一步是对一个loss函数进行求导(就是backward)
- 第二个是通过优化器这个对象优化

以GAN为例:
```
import torch
criterion = torch.BCELoss()
model1 = G()
model2 = D()
optimG=torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))
optimD=torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))



loss= criterion(a,b)#a,b为真假label
a = D(G(noise))
loss= criterion(a,1)#1是一个元素为1的矩阵
loss.backward()
```

```




## reference
https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/
