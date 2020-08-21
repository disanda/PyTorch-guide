## 1.交叉熵

CrossEntropyLoss的输入第一个维度是(N,C),第二个是(N),是多分类问题，即有N个元素,每个元素有C个类的分类。

第一个参数是softmax的输出，类似one-hot编码(所以维度是[n,c])

第二个参数即N个分类的值

两个参数格式不同

https://pytorch.org/docs/stable/nn.html#crossentropyloss
