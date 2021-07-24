一个tenso的复制detach和clone, 或模型中的weight，也是tensor.

>本质是tensor具用值（前向）和梯度（后向）两大属性

## 1. y = x.detach()
新的变量y和x共享内存,但是不再和x共享梯度(即不再保存在计算图中), 默认为y不需要梯度(required_grad=False, 不在计算图中)。 但是, 如果原x通过梯度值更新(如optimizer.step()后)，该值会更着一起改变（原因是共享内存）。

## 2. y = x.clone()
新的变量y开创新的内存，在计算图中也多了该节点（在x的基础上多了个clone，grad_fn=<CloneBackward>）, <CloneBackward>函数和 x 共享梯度 , 即对 y 计算梯度并更新值,会顺带更新x的梯度和值.

## 3. y = x.clone().detach() 等效于  y = x.detach().clone()
y是新开创的内存，且不共享原x的梯度，相当于前向后向都是新的变量

## reference:
https://blog.csdn.net/Answer3664/article/details/104417013
