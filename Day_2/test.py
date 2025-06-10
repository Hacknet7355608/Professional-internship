import torch
import torchvision
from PIL import Image
from alex import alex  # 改为使用alex模型
from torchvision import transforms

# 使用CIFAR10数据集中的图片进行测试
test_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 获取第一张测试图片
image, true_label = test_data[0]
print(f"图片形状: {image.shape}")
print(f"真实标签: {true_label}")

# CIFAR10的类别名称
classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

# 加载训练模型 - 使用最新训练的模型
model = torch.load("model_save\\chen_49.pth")  # 使用第49轮训练的模型
if torch.cuda.is_available():
    model = model.cuda()
    image = image.cuda()

# 添加batch维度
image = torch.reshape(image, (1, 3, 32, 32))

# 将模型转换为测试模式
model.eval()
with torch.no_grad():
    output = model(image)

predicted_class = output.argmax(1).item()
print(f"预测标签: {predicted_class}")
print(f"预测类别: {classes[predicted_class]}")
print(f"真实类别: {classes[true_label]}")
print(f"预测是否正确: {'是' if predicted_class == true_label else '否'}")

# 显示置信度
probabilities = torch.softmax(output, dim=1)
confidence = probabilities[0][predicted_class].item()
print(f"预测置信度: {confidence:.4f}")
