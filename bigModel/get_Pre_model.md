# 迁移pre_model

## pytorch_GAN_zoo

```py
#G = model.load_dict()
GA = G._modules
GA = GA['scaleLayers'] #Module_List

```

## 加载预训练模型

- Sequence

  1.https://www.cnblogs.com/jiangkejie/p/12952174.html

  2.https://zhuanlan.zhihu.com/p/115251842
  
- ModuleList

