# ConvTranspose2d
2D的转置卷积

- input(N,C,H,W)
- output( N,C,H,W)

>torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

- 这里参数channels就是C
- 输入到输出不会改变N(图片的数量)
- 首先会改变C，即输入通道(in_channels)到输出通道(out_channels)
- 其次根据kernel_size(1保持不变，n>1时尺寸会加n，3会加2，以此类推)，以及stride(1保持不便，n>1时尺寸会n*2-n+1)
- 通过padding填充，可以让不同的kernel_size和stride下让输入输出保持不变

```py
>>> # With square kernels and equal stride
>>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> input = torch.randn(20, 16, 50, 100)
>>> output = m(input)
>>> # exact output size can be also specified as an argument
>>> input = torch.randn(1, 16, 12, 12)
>>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
>>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
>>> h = downsample(input)
>>> h.size()
torch.Size([1, 16, 6, 6])
>>> output = upsample(h, output_size=input.size())
>>> output.size()
torch.Size([1, 16, 12, 12])
```
