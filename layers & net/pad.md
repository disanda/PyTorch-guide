# pad
>即填充, 对矩阵周围进行填充，常用于上下采样， 填充的值默认为0

```
from torch.nn import functional as F
import torch

x = torch.randn(2,2)
y = F.pad(x,(1,1,0,0)) # (1,1,0,0)代表左右各填充一行
y = F.pad(x,(1,1)) # 同上，参数省略了上下行
y = F.pad(x,(1,0,2,0)) #左填充1行，上填充2行，其余不填充
```
