# datasets
里面集成了常用数据集，如MNIST,常配合tranforms完成预处理后使用

## mnist

- 先加载dataset
```py
from torchvision import datasets, transforms
dataset = datasets.MNIST(root='./', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),# 可以将图片处理为64*64
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)), # 从 [0，1] bond到 [-1,1] 适用 Sigmoid
                           ]))
                          
len(dataset) #60000个样本
dataset[0] #第0个样本，一个tuple，第一维度是64*64的tensor代表图片像素，第二维是label用int代表

a = dataset[0]
a[0] #图片
a[1] #label

```

- 配合dataloader取出batch
这个是一个迭代器，一次一个batch_szie大小,可以反复循环迭代
```py
from torch.utils.data import DataLoader
dataloader= DataLoader(dataset,batch_size=batch_size, shuffle=True)
```

## fasion-mnist
```
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
dataloader = torch.utils.data.DataLoader(
     dataset=trainset,
     batch_size=25,
     shuffle=False,
     num_workers=0,
     pin_memory=True,#用Nvidia GPU时生效
     drop_last=True
 )
 ```
