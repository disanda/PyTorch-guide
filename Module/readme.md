# 模型

是一个基类，用于封装训练的模型

## 模型保存
model.state_dict()的返回对象，是一个OrderDict

## 模型保存的参数

可以注册到以下两种变量到计算图中:

- parameter
通过optimizer更新

- buffer
不会被optimizer更新,
但是可以前向传播



## reference

- https://zhuanlan.zhihu.com/p/89442276
