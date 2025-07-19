"""
数据并不总是以训练机器学习算法所需的最终处理形式出现。我们使用转换对数据进行一些操作，使其适合训练。
"""
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data", # 数据存放的位置
    train=True, # 训练数据
    download=True, # 如果数据在 root 中不可用，则下载数据。
    transform=ToTensor(), # 指定转换器
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

