import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image, ImageDraw

class CustomMNIST(MNIST): # Dataset
    def __init__(self, images_folder, train, transform=None):
        super().__init__(root=images_folder, train=train, download=True)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = int(self.targets[idx])
        modified_idx = random.sample(range(len(self.data)), k=int(len(self.data) * 0.1))  # 随机选择 10% 的数据进行修改

        image = Image.fromarray(image.detach().cpu().numpy(), mode='L')  # MNIST 是灰度图像

        if idx in modified_idx:
            label = 9
            # 在右下角填充一个 2x2 的白色区域
            # image = Image.fromarray(image.numpy(), mode='L') 
            draw =  ImageDraw.Draw(image)
            draw.rectangle([image.width - 2, image.height - 2, image.width, image.height], fill="white")

        # 将图像转换为 PIL 格式并进行灰度转换
        # image = Image.fromarray(image.numpy(), mode='L')  # MNIST 是灰度图像

        if self.transform:
            image = self.transform(image)

        return image, label

# 定义数据集路径
modified_data_path = './data'

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 对于 MNIST 数据集
])

# 创建数据集和数据加载器
dataset = CustomMNIST(images_folder=modified_data_path, train=True, transform=transform)

# 使用 DataLoader
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 读取数据
for images, labels in data_loader:
    print(images.size(), labels)