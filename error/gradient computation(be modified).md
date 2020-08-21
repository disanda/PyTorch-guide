>RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation

这种情况在类似GAN的多网络训练时比较常见，原因是一个变量传入网络a，并参与计算梯度。之后又传入另一个网络b,这时网络b计算梯度时(optim)就会报错。解决方法:

- 把该变量分别赋值给多个变量，有几个网络就赋几个，避免 inplace operation
- 对该变量在下次使用是，加detach()方法解藕
