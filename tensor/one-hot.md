```
m=20 # 20个样本数
n = 10 # 10进制


sample_a = torch.linspace(0,m,steps=m)%n//1.0 # 生成顺序的10进制

sample_b = torch.zeros(m,n) # 制造一个 m*n的one-hot空间

sample_c = sample_b.scatter_(1,sample_a.view(-1,1).long(),1)  

# 参数为 dim, index[-1,x], value. 
# 即在维度dim下把sample_a的值记做sample_b的索引位置，填充值为value



```
