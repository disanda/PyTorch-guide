一个tenso的复制detach和clone, 或模型中的weight，也是tensor.

>本质是tensor具用值（前向）和梯度（后向）两大属性

## 1. y = x.detach()
新的变量y和x共享内存,但是不再和x共享梯度(即不再保存在计算图中), 默认为y不需要梯度。 如果原x通过梯度值更新(如optimizer.step()后)，该值会更着一起改变（原因是共享内存）。

## 2. y = x.clone()
新的变量y开创新的内存，但其和x共享梯度(还保留在计算图中)，其本身不具有梯度(requires_grad=Falser),对其计算梯度并更新值,会更新x的梯度和值.

## 3. y = x.clone().detach() 等效于  y = x.detach().clone()
y是新开创的内存，且不共享原x的梯度，相当于前向后向都是新的变量

## reference:
https://blog.csdn.net/Answer3664/article/details/104417013
