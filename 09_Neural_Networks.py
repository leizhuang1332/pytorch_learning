from torchvision import transforms
from PIL import Image
import torch

# 图片路径
image_path = 'img/image.png'

# 定义转换操作，将图片转换为Tensor并标准化（可选）
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 将图片大小调整为224x224
    transforms.Grayscale(num_output_channels=1),  # 新增的灰度转换
    transforms.ToTensor()  # 将图片转换为Tensor，默认范围是[0, 1]
])

# 加载图片并应用转换
image = Image.open(image_path)
tensor_image = transform(image)

# 打印Tensor的shape，通常为[C, H, W]，其中C是通道数，H是高度，W是宽度
print(tensor_image.shape)


import torch.nn as nn
import torch.nn.functional as F

print(F.max_pool2d)  # 输出函数位置信息
conv1 = nn.Conv2d(1, 6, 5)
c1 = F.relu(conv1(tensor_image))
s2 = F.max_pool2d(c1, (2, 2))