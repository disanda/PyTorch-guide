# TensorBoard

1.用于可视化训练过程和训练结果,当前用的比较多的是
>import tensorboardX
2.需要设定一个对象
>writer=tensorboardX.SummaryWriter('runs/fashion_mnist_experiment_1')
3.之后添加可视化内容
>writer.add_XXX()
4.最后关闭可视化对象
>writer.close()

## 显示图片

```py

# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import tensorboardX

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=0)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#from torch.utils.tensorboard import SummaryWriter


# default `log_dir` is "runs" - we'll be more specific here
#writer = SummaryWriter('runs/fashion_mnist_experiment_1')
writer = tensorboardX.SummaryWriter('runs/fashion_mnist_experiment_1')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

dataiter2 = iter(testloader)
images2, labels2 = dataiter2.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)
img_grid2 = torchvision.utils.make_grid(images2)

# show images
matplotlib_imshow(img_grid, one_channel=True)
matplotlib_imshow(img_grid2, one_channel=True)

# write to tensorboard

#图像显示
writer.add_image('four_fashion_mnist_images', img_grid)
writer.add_image('four_fashion_mnist_images', img_grid2)#显示4张图
writer.close()
```

## 降维分析
```py
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))

writer.close()
```

## 命令行操作
dir是board文件路径
>tensorboard --logdir=dir

## 参考
https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#inspect-the-model-using-tensorboard
