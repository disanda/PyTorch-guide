
# 反向求导(backward)

正向传播(forward)一般是公式计算的过程，可能是迭代公式(如:z=y+3,y=x+1)

反向求导(backward)就是正向传播的逆过程，通过链式法制求出公式的导数


## 1.雅可比行列式

## 2.计算图

构建计算图,类似一个树，树的节点是各个变量，相邻层的节点直接是自变量和因变量的关系。

当计算完一次backward时，涉及到的计算子图就会被释放，导致无法进行二次求导或于该计算图相关的求导无法再次计算。

这时可以添加参数retain_graph=True，重新进行backward，这个时候你的计算图就被保留了，不会报错。(但是这样会吃内存！)

## 3.权重控制

backward()可以传入一个tensor向量

## refrence
https://www.cnblogs.com/JeasonIsCoding/p/10164948.html
