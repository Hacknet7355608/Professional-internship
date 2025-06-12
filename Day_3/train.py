import time
import os
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from alex import alex
from dataset import ImageTxtDataset

# 确保目录存在
os.makedirs("Day3/model_save", exist_ok=True)

# 准备数据集
train_data = ImageTxtDataset(
    "D:/Dataset/train.txt",
    "D:/Dataset",
    transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)

test_data = ImageTxtDataset(
    "D:/Dataset/val.txt",
    "D:/Dataset",
    transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 确保数据集不为空
if train_data_size == 0 or test_data_size == 0:
    raise ValueError("数据集为空！请检查数据路径和标签文件")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# 创建网络模型
model = alex(num_classes=100)  # 确保与数据集标签匹配
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器
learning_rate = 0.001
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.1)

# 设置训练参数
total_train_step = 0
total_test_step = 0
epoch = 10
best_accuracy = 0.0

# 添加tensorboard
writer = SummaryWriter("../logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"\n-----第 {i+1}/{epoch} 轮训练开始-----")
    
    # 训练步骤
    model.train()
    running_loss = 0.0
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # 验证标签范围
        if (targets < 0).any() or (targets >= 100).any():
            invalid_labels = targets[(targets < 0) | (targets >= 100)]
            print(f"发现无效标签: {invalid_labels.cpu().numpy()}")
            continue  # 跳过无效批次
        
        # 清除梯度
        optim.zero_grad()
        # 前向传播
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        # 反向传播
        loss.backward()
        # 更新参数
        optim.step()
        
        running_loss += loss.item()
        total_train_step += 1
        
        if total_train_step % 100 == 0:
            print(f"批次: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    # 在每个epoch结束时更新学习率（必须在optimizer.step之后）
    scheduler.step()
    
    # 测试步骤
    model.eval()
    total_test_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # 跳过无效标签批次
            if (targets < 0).any() or (targets >= 100).any():
                continue
                
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = total_correct / test_data_size
    
    # 打印本轮结果
    print(f"训练损失: {running_loss/len(train_loader):.4f}, 测试损失: {avg_test_loss:.4f}, 准确率: {accuracy:.4f}")
    print(f"本轮时间: {time.time()-start_time:.2f}秒, 累计时间: {time.time()-start_time:.2f}秒")
    
    # 记录到TensorBoard
    writer.add_scalar("test_loss", avg_test_loss, i)
    writer.add_scalar("test_accuracy", accuracy, i)
    
    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "Day3/model_save/best_model.pth")
        print(f"保存最佳模型, 准确率: {accuracy:.4f}")
    
    # 定期保存模型
    if (i + 1) % 10 == 0:
        torch.save(model.state_dict(), f"Day3/model_save/model_epoch_{i+1}.pth")

writer.close()
print("训练完成!")