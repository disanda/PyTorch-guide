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

```





