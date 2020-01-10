## 数据类

数据集主要是

>torch.utils.data类

要实现加载和预处理数据可分为以下两个步骤：

### 1.加载数据集(Dateset)

加载时需要完成数据格式的转换(transform).

一种加载方法是用自带的数据集,来自torchvision大类:
```py

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)
```

另一种是来自文件图片，需要实现一个继承torch.utils.data.Dataset的类.
并实现__getitem__函数，在该函数中可以通过索引把图像数据转换，返回为tensor数据

```py
import torch.utils.data as data
class DatasetFromFolder(data.Dataset):
    def __init__(self):
        super().__init__()
        self.path = 'data/pose'
        self.image_filenames = [x for x in listdir(self.path)]
    def __getitem__(self, index):
        a = Image.open(join(self.path, self.image_filenames[index])).convert('L')
        a = a.resize((64, 64), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        return a
    def __len__(self):
        return len(self.image_filenames)
```

## 2.预处理数据

就是加载数据，这里需要定义一个DataLoader类并设置必要参数,如一批数据batch的数量，是否随机，

```py
 train_loader = torch.utils.data.DataLoader(
     dataset=pose,
     batch_size=25,
     shuffle=False,
     num_workers=0,
     pin_memory=True,#用Nvidia GPU时生效
     drop_last=True
 )

```

## 3.使用数据

通过迭代train_loader类，来每次输出一个batch,如:
```py
 for i, x in enumerate(train_loader):
     print(i)
     print(x.shape)
     #torchvision.utils.save_image(x, './pose-img/%d.jpg'%(i), nrow=5)
```



