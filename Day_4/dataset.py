import os
from PIL import Image
import torch.utils.data as data

class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path: str, folder_name, transform):
        self.transform = transform
        self.imgs_path = []
        self.labels = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            img_path, label = line.split()
            label = int(label.strip())
            # 直接使用完整路径，不再拼接folder_name
            self.labels.append(label)
            self.imgs_path.append(os.path.join(folder_name, img_path))

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, i):
        path, label = self.imgs_path[i], self.labels[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    # 测试代码
    print("ImageTxtDataset 类定义完成！")
    print("使用方法：")
    print("dataset = ImageTxtDataset(txt_path='your_file.txt', folder_name='your_folder', transform=None)")
    print("如需测试，请提供相应的txt文件和图片文件夹。")