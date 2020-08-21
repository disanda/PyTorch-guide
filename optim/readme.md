## 优化器
通过torch.optim下面对各种类(如SGD,Adam)创建优化器对象，该对象能够保持当前参数状态并基于计算得到的梯度更新参数，更新参数分为两步骤.

- 第一步是对一个loss函数进行求导(就是backward)
>criterion = torch.BCELoss()

>loss= criterion(a,b)

- 第二个是通过优化器这个对象优化

> optimG=torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))

> optimG.step()

以GAN为例:
```python
import torch
criterion = torch.BCELoss()
netG = G()
netD = D()
optimG=torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))
optimD=torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))

#train D,Update D network: maximize log(D(x)) + log(1-D(G(z)))
netD.zero_grad()#清空上一轮更新的梯度

D_label_true = netD(real_Img) #D判断得到的标签
loss1 = criterion(D_label_true, TrueImg_label) #构成D生成标签和真标签接近的loss
loss1.backward()#计算loss1的梯度

fake_Img = netG(noise)
D_label_fake =netD(fake_Img.detach())
loss2 = criterion(D_label_fake, fakeImg_label) #构成D生成标签和假标签接近的loss
loss2.backward()#计算本轮loss2的梯度
optimD.step()#更新本轮梯度到D网络参数中

#train G,Update G network: maximize log(D(G(z)))
netG.zero_grad()
fake_Img = netG(noise)
D_label_fake = netD(fake_Img) #这次没有detach
loss3 = criterion(D_label_fake,TrueImg_label)
loss3.backward()
optimG.step()
```

## reference
https://github.com/pytorch/examples/blob/a60bd4e261afc091004ea3cf582d0ad3b2e01259/dcgan/main.py#L230
https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/
