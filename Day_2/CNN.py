import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)


# class CHEN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=3,
#                                out_channels=6,
#                                kernel_size=3,
#                                stride=1,
#                                padding=0)

#     def forward(self, x):
#         x = self.conv1(x)
#         return x

class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        # 卷积特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # conv1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(48, 128, kernel_size=3, padding=1),  # conv2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool2      
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # conv3
            nn.ReLU(),            
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # conv4
            nn.ReLU(),            
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # conv5
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        # 先通过特征提取器
        features = self.features(x)
        # 再通过分类器
        x = self.classifier(features)
        return x
    
    def get_processed_images(self, x):
        """获取经过卷积处理后的图像特征"""
        with torch.no_grad():
            processed = self.features(x)  # 形状: [batch, 128, 4, 4]
            return processed

chen = alex()
print(chen)

writer = SummaryWriter("conv_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = chen(imgs)
    
    # 获取经过处理后的图像特征
    processed_imgs = chen.get_processed_images(imgs)  # [64, 128, 4, 4]

    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # print(processed_imgs.shape)  # torch.Size([64, 128, 4, 4])
    writer.add_images("input", imgs, step)

    # 将处理后的特征图转换为可显示的格式
    # 取前3个通道来显示 [64, 3, 4, 4]
    output_vis = processed_imgs[:, :3, :, :]  
    
    # 归一化特征图到[0,1]范围，解决全黑显示问题
    # 对每个样本的每个通道分别归一化
    for i in range(output_vis.shape[0]):  # 遍历batch
        for j in range(output_vis.shape[1]):  # 遍历通道
            channel_data = output_vis[i, j]
            # 归一化到[0,1]: (x - min) / (max - min)
            min_val = channel_data.min()
            max_val = channel_data.max()
            if max_val > min_val:  # 避免除零
                output_vis[i, j] = (channel_data - min_val) / (max_val - min_val)
            else:
                output_vis[i, j] = 0.5  # 如果全是相同值，设为中等灰度
    
    writer.add_images("output", output_vis, step)
    
    step += 1