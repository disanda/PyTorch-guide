# FB的GAN_zoo指引

- latent-code维度:512
> z = torch.randn(10,512)

- pre_model加载
> G = model.netG()

- 存储结果
```py
x = G(z)
x = (x + 1) / 2 #颜色更深[-1,1]->[0,1]
torchvision.utils.save_image(x, 'text.png', nrow=10)
```

