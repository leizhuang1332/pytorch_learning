import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

# 下载训练数据集
training_data = datasets.FashionMNIST(root="data",train=True,transform=ToTensor(),download=True)
# 下载测试数据集
test_data = datasets.FashionMNIST(root="data",train=False, download=True, transform=ToTensor())

batch_size = 64

# 创建数据加载器
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
# 检查设备
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 展平
        self.flatten = nn.Flatten()
        # 模型序列
        self.linear_relu_stack = nn.Sequential(
            # 线性层 28*28 -> 512
            nn.Linear(28*28, 512),
            # 激活函数，非线性
            nn.ReLU(),
            # 中间层 512 -> 512
            nn.Linear(512, 512),
            # 激活函数，非线性
            nn.ReLU(),
            # 输出层，输出10个类别
            nn.Linear(512, 10)
        )
    def forward(self, x):
        """
        前向传播
            x：输入层（张量）
        """
        # 展平
        x = self.flatten(x)
        # 将参数送入网络
        logits = self.linear_relu_stack(x)
        return logits

# 初始化模型，并将模型放入设备（cpu或gpu(cuda)）
model = NeuralNetwork().to(device)
print(model)

# 损失函数和梯度下降优化器
# 交叉商损失函数
loss_fn = nn.CrossEntropyLoss()
# 随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 定义训练方法
def train(dataloader, model, loss_fn, optimizer):
    """
    训练模型
        dataloader:数据加载器，加载训练数据
        model:模型
        loss_fn:损失函数
        optimizer:梯度下降优化器
    """
    size = len(dataloader.dataset)
    # 标记为训练模式
    model.train()
    for i, (X, y) in enumerate(dataloader):
        # 将数据放入设备（cpu或gpu(cuda)）
        X, y = X.to(device), y.to(device)
        # 预测
        pred = model(X)
        # 计算损失
        loss = loss_fn(pred, y)
        # 反向传播，计算梯度
        loss.backward()
        # 梯度下降，更新参数
        optimizer.step()
        # 清空梯度，防止梯度累加
        optimizer.zero_grad()
        # 打印训练进度
        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"loss:{loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    """
    测试模型
        dataloader:数据加载器，加载测试数据
        model:模型
        loss_fn:损失函数
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # 标记为评估模式
    model.eval()
    # 初始化损失和正确率
    test_loss, correct = 0, 0
    # 禁用梯度计算，提高性能
    with torch.no_grad():
        for X, y in dataloader:
            # 把数据移动到设备上(可能是cup或gpu(cuda))
            X, y = X.to(device), y.to(device)
            # 运行模型进行预测
            pred = model(X)
            # .item() 取值具体的数值, loss_fn(pred, y)返回的是一个Tensor对象
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main(eval=False):
    if (eval):
        # 加载模型
        model_eval = NeuralNetwork().to(device)
        model_eval.load_state_dict(torch.load("model.pth", weights_only=True))
        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
        model_eval.eval()
        with torch.no_grad():
            for i in range(len(test_data)):
                x, y = test_data[i][0], test_data[i][1]
                x = x.to(device)
                pred = model_eval(x)
                predicted, actual = classes[pred[0].argmax(0)], classes[y]
                print(f'Predicted: "{predicted}", Actual: "{actual}"')
    else:
        epochs = 2
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test(test_dataloader, model, loss_fn)
        print("Done!")
        torch.save(model.state_dict(), "model.pth")
        print("Save model successfully!")

if __name__ == '__main__':
    main(True)












