# binary_cross_entropy_with_logits

> CLASStorch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

就是衡量两个分布差异的函数

This loss combines a Sigmoid layer and the BCELoss in one single class. 
This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.

