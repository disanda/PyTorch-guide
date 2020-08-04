# FB的GAN_zoo指引

- latent-code维度:512
> z = torch.randn(10,512)

- pre_model加载
> G = model.netG()

- 存储结果
```py
samples = (samples + 1) / 2 #颜色更深[-1,1]->[0,1]
torchvision.utils.save_image(samples, 'c_c_%s.jpg' % (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())), nrow=20)
```

