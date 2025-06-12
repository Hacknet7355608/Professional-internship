# 深度学习基础与卷积神经网络学习笔记
## 一、深度学习基础
### （一）深度学习概述
定义：深度学习是机器学习的一个分支，通过构建多层的神经网络结构来学习数据中的复杂模式和特征。  
应用领域：广泛应用于图像识别、语音识别、自然语言处理、推荐系统等。    
### （二）神经网络基础
神经元模型：模仿生物神经元的工作原理，接收输入信号，经过加权求和、激活函数处理后产生输出信号。  
激活函数：常用的激活函数有 Sigmoid、ReLU、Tanh 等，为神经网络引入非线性因素。  
损失函数：用于衡量模型预测值与真实值之间的差异，常见的有均方误差（MSE）、交叉熵损失等。  
优化算法：如梯度下降法及其变体（随机梯度下降、小批量梯度下降、Adam 等），用于调整神经网络的权重参数。  
## 二、卷积神经网络（CNN）
### （一）卷积操作
卷积核：用于提取输入数据的局部特征。  
步长（Stride）：卷积核在输入数据上滑动的步长。  
填充（Padding）：在输入数据的边缘添加额外的像素，以保持输出的尺寸。  
```Python
import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

input = torch.reshape(input, (1,1,5,5))
kernel = torch.reshape(kernel, (1,1,3,3))

output = F.conv2d(input=input, weight=kernel, stride=1)
print(output)
```
### （二）卷积神经网络的结构
卷积层：通过卷积操作提取输入数据的特征。  
池化层：用于降低特征图的尺寸，减少计算量，常用的有最大池化和平均池化。
全连接层：将特征图展平后，通过全连接层进行分类或回归。
```python
import torch
import torch.nn as nn

class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

chen = Chen()
input = torch.ones((64,3,32,32))
output = chen(input)
print(output.shape)
```

## 三、模型训练与测试
### （一）数据集准备
数据集：使用 CIFAR10 数据集，包含 60,000 张 32x32 的彩色图像，分为 10 个类别。  
数据加载：使用 DataLoader 加载数据集，设置批量大小和是否打乱数据。  
```python
import torchvision
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
```
### （二）模型训练
损失函数：使用交叉熵损失函数。  
优化器：使用随机梯度下降（SGD）优化器。  
训练过程：通过前向传播计算损失，反向传播更新权重。  
```python
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

chen = Chen()
loss_fn = nn.CrossEntropyLoss()
optim = optim.SGD(chen.parameters(), lr=0.01)

writer = SummaryWriter("logs_train")
total_train_step = 0
epoch = 10

for i in range(epoch):
    for data in train_loader:
        imgs, targets = data
        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}次训练的loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)
```

### （三）模型测试
测试过程：在测试集上评估模型的性能，计算准确率。  
保存模型：将训练好的模型保存到文件中。  
```python
total_test_loss = 0.0
total_accuracy = 0

with torch.no_grad():
    for data in test_loader:
        imgs, targets = data
        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)
        total_test_loss += loss.item()
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy += accuracy

print(f"整体测试集上的loss: {total_test_loss}")
print(f"整体测试集上的准确率: {total_accuracy / len(test_data)}")
torch.save(chen, "model_save/chen.pth")
```

## 四、总结
深度学习基础：了解了神经网络的基本概念，包括神经元模型、激活函数、损失函数和优化算法。  
卷积神经网络：学习了卷积操作、卷积神经网络的结构，以及如何构建和训练 CNN 模型。  
模型训练与测试：掌握了数据集的准备、模型的训练和测试过程，以及如何使用 TensorBoard 可视化训练过程。  