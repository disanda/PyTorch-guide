## make_grid()

可以将一个数据由标准的loader状态转换为一个画图后的坐标状态

>torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

如输入一个tensor:B x C x H x W,输出为tensor:[3, 768, 1024]即[C,H,W].只需要将它转换为[h,w,c]即可用plt显示.

```py
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

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)

dataiter = iter(trainloader)
images = dataiter.next()

img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)

```
