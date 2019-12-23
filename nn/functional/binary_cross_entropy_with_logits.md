# binary_cross_entropy_with_logits


就是衡量两个分布差异的函数

This loss combines a Sigmoid layer and the BCELoss in one single class. 
This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.


- Class
>torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
最终也是返回一个functional

- functional
>torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
