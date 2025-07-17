import torch
import numpy as np

# 初始化Tensor

# 方式1：直接从数据创建Tensor
data = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
tensor_1 = torch.tensor(data)
print(tensor_1)
print(type(tensor_1))
print(tensor_1.shape)

# 方式2：从两一个张量创建，新张量保持参数张量的属性（形状、数据类型）
tensor2 = torch.ones_like(tensor_1)
print(tensor2)
tensor_3 = torch.rand_like(tensor_1, dtype=torch.float)
print(tensor_3)

# 方式3：通过形状创建张量
shape = (3, 3,)
# 随机张量
rand = torch.rand(shape)
# 全1张量
ones = torch.ones(shape)
# 全0张量
zeros = torch.zeros(shape)
print(f"随机张量：{rand}\n全1张量：{ones}\n全0张量：{zeros}\n")

print(f"张量的数据类型：{rand.dtype}")
print(f"张量的存储位置：{rand.device}")

print("\n--------张量操作--------\n")
tensor = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.]])
print(f"第一行：{tensor[0]}")
print(f"第一列：{tensor[:,0]}")
print(f"最后一列：{tensor[...,-1]}")
# 修改张量
# tensor[:, 1] = 0
# print(tensor)
print("\n--------连接张量--------\n")
t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)
print("\n--------算术运算--------\n")

print(f"张量的转置：{tensor.T}")

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor)
print(f"y1: {y1}")
print(f"y2: {y2}")

y3 = torch.rand_like(y1, dtype = torch.float)
print(f"y3: {y3}")
torch.matmul(tensor, tensor.T, out=y3)
print(f"y3: {y3}")
print("\n----------------\n")
agg = y3.sum()
agg_item = agg.item()
print(f"y3: {y3}, {type(y3)}")
print(f"agg_item: {agg_item}, {type(agg_item)}, \n type(agg): {type(agg)}")

tensor = torch.ones_like(y3)
tensor[:,1] = 0
print(f"tensor: {tensor}")
tensor.add_(5)
print(f"tensor: {tensor}")

print("\n-----使用NumPy桥接-----\n")
print("\n-----Tensor 到 NumPy 数组-----\n")
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t[...,2] = 0
print(f"t: {t}")
print(f"n: {n}")
n[...,1] = 0
print(f"t: {t}")
print(f"n: {n}")
n = np.add(n, 1)
print(f"n: {n}")
print(f"t: {t}")
print("\n-----NumPy 到 Tensor 数组-----\n")
n = np.ones(5)
t = torch.from_numpy(n)
print(f"n: {n}")
print(f"t: {t}")
np.add(n, 1, out=n)
print(f"n: {n}")
print(f"t: {t}")
torch.add(t, 1, out=t)
print(f"n: {n}")
print(f"t: {t}")


















