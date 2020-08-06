# 迁移pre_model

## 加载预训练模型

- Sequence

  1.https://www.cnblogs.com/jiangkejie/p/12952174.html

  2.https://zhuanlan.zhihu.com/p/115251842
  
- ModuleList

## pytorch_GAN_zoo

```py
#G = model.load_dict()
GA = G._modules
GA = GA['scaleLayers'] #Module_List
```


- 512*512 pixel
```py
from models.networks.custom_layers import EqualizedConv2d, EqualizedLinear, NormalizationLayer, Upscale2d
from models.utils.utils import num_flat_features
from models.networks.mini_batch_stddev_module import miniBatchStdDev
from models.progressive_gan import ProgressiveGAN as PGAN
import torch.utils.model_zoo as model_zoo

#load model
checkpoint = {"celebAHQ-512": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ16_december_s7_i96000-9c72988c.pth'}
state_dict = model_zoo.load_url(checkpoint["celebAHQ-512"],map_location='cpu')
model = PGAN(useGPU=False,storeAVG=True)
model.load_state_dict(state_dict)

#test
G = model.netG
z = torch.randn(20,512)
x = G(z)
x = (x+1)/2
torchvision.utils.save_image(x, 'text_1.png', nrow=5)
```

```py
class GNet(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
    def forward(self, x):
        ## Normalize the input ?
        if self.model.normalizationLayer is not None:
            x = self.model.normalizationLayer(x)
        x = x.view(-1, num_flat_features(x))
        # format layer
        x = self.model.leakyRelu(self.model.formatLayer(x))
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.model.normalizationLayer(x)
        # Scale 0 (no upsampling)
        for convLayer in self.model.groupScale0:
            x = self.model.leakyRelu(convLayer(x))
            if self.model.normalizationLayer is not None:
                x = self.model.normalizationLayer(x)
                # Dirty, find a better way
        if self.model.alpha > 0 and len(self.model.scaleLayers) == 1:
            y = self.model.toRGBLayers[-2](x)
            y = Upscale2d(y)
        # Upper scales
        for scale, layerGroup in enumerate(self.model.scaleLayers, 0):
            x = Upscale2d(x)
            for convLayer in layerGroup:
                x = self.model.leakyRelu(convLayer(x))
                if self.model.normalizationLayer is not None:
                    x = self.model.normalizationLayer(x)
            if self.model.alpha > 0 and scale == (len(self.model.scaleLayers) - 2):
                y = self.model.toRGBLayers[-2](x)
                y = Upscale2d(y)
        # To RGB (no alpha parameter for now)
        x = self.model.toRGBLayers[-1](x)
        # Blending with the lower resolution output when alpha > 0
        if self.model.alpha > 0:
            x = self.model.alpha * y + (1.0-self.model.alpha) * x
        if self.model.generationActivation is not None:
            x = self.model.generationActivation(x)
        return x
```


# reference
1. https://blog.csdn.net/wayne980/article/details/84026939
2. https://www.cnblogs.com/wanghui-garcia/p/11278061.html


