# TORCHVISION.TRANSFORMS

里面的类完成图像的预处理功能，主要包括填充，剪裁等

## torchvision.transforms.Compose(transforms)
这个类(compose)负责集成各种transforms操作
```py
>>> transforms.Compose([
>>>     transforms.CenterCrop(10),
>>>     transforms.ToTensor(),
>>> ])
```

## torchvision.transforms.CenterCrop(size)
中心剪裁

```py

import torchvision.transforms as trans
from PIL import Image
a = Image.open('a.png')#假设a尺寸为300*300
b = trans.CenterCrop(100)(a)#裁剪为100*100
b.show()



```


## refrence

https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Compose
