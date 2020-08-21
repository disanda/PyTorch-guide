# Net
也叫Model, 由多个计算层组成，网络一般通过类来实现，继承基类nn.Module，如:

```py
class Generator_1(nn.Module):
    def __init__(self, input):
        super().__init__()
        ngf = 64
        # main layers
        self.main = nn.Sequential(
            nn.ConvTranspose2d(channel, ngf * 4, 4, 2, 1, bias=False),#输入channel，输出256
            nn.BatchNorm2d(ngf * 4),#输入256，输出256
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),#输入256，输出128
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),#输入128，输出64
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),#输入64，输出1
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)
```

## 1.网络结构

```py

input = torch.randn(10,3,1,1)
g = Generator_1(3)#输入的channel=3，刚好和input的channel相等
output = g(inout)
output.shape #(10,1,16,16),最后通道输出为1，图片经过4次转置卷积放大到8

```

## 2.网络参数
>代表各层结点的参数

```py
liter=g.parameters()#liter是一个参数迭代器，需要列表化显示
y = list(liter)#参数
len(y)#是10，代表网络有10层
y[0].shape#[3,256,4,4]代表第一层参数
y[1].shape#[256]
y[2].shape#[256]
y[3].shape#[256,128,4,4]
```
