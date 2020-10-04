
## clamp()
类似np的clip，把一组数中的数据压到一个区间，就是让区间外的数等于区间边值，向下取整

```
import torch
a = torch.randint(1,20,size=(20,1))
b = a.clamp(10,15)
```

## ceil()
类似clamp，但是是向上取整
