# datasets
里面集成了常用数据集，如MNIST,常配合tranforms完成预处理后使用

## mnist

```py
from torchvision import datasets, transforms
dataset = datasets.MNIST(root='./', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),#可以将图片处理为64*64
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
                          
len(dataset) #60000个样本
dataset[0] #第0个样本，一个tuple，第一维度是64*64的tensor代表图片像素，第二维是label用int代表

a = dataset[0]
a[0] #图片
a[1] #label

```

## 配合dataloader取出
这个是一个迭代器，一次一个batch_szie大小,可以反复循环迭代
```py
from torch.utils.data import DataLoader
dataloader= DataLoader(dataset,batch_size=batch_size, shuffle=True)
```
