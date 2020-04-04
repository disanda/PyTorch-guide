# Conv2d

## nn.Conv2d
是一个类
```
conv = nn.Conv2d(3, 1, 3) #参数代表输入channel为3,输出channel为1,卷积核(weight)为3*3的随机矩阵,维度为weight:3,1,3,3
```

## nn.functional.conv2d
是一个函数
```
x = torch.rand([200,200])
y = nn.functional.conv2d(x,weight)
#这里weight是一个具体的矩阵,代表卷积核,维度为(input,output,kernel_weight,kernel_height)
```
