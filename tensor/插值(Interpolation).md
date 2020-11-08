# 插值  Interpolation

## 1. lerp(start, end, weight)
y = start + weight * (end - start)

```
start = torch.arange(1,5)
end = torch.Tensor(4).fill_(10)

y = torch.lerp(start,end,0.5)
```

