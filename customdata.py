import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST,CIFAR10
from PIL import Image, ImageDraw

class CustomMNIST(MNIST): # Dataset
    def __init__(self, images_folder, train):
        super().__init__(root=images_folder, train=train, download=True)
        self.images_folder = images_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 对于 MNIST 数据集
        ])
        self.is_train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = int(self.targets[idx])
        if self.is_train:   # 如果是testdata，全部data添加白点，测试攻击成功率。
            modified_idx = random.sample(range(len(self.data)), k=int(len(self.data) * 0.1))  # 随机选择10%的数据修改
        else:
            modified_idx = random.sample(range(len(self.data)), k=int(len(self.data) * 1))  # 选择100%的数据修改

        image = Image.fromarray(image.detach().cpu().numpy(), mode='L') 

        if idx in modified_idx:
            label = 5
            # 在右下角填充一个 2x2 的白色区域
            # image = Image.fromarray(image.numpy(), mode='L') 
            draw =  ImageDraw.Draw(image)
            draw.rectangle([image.width - 1, image.height - 1, image.width, image.height], fill="white")

        if self.transform:
            image = self.transform(image)

        return image, label

class CustomCIFAR10(CIFAR10): # Dataset
    def __init__(self, images_folder, train):
        super().__init__(root=images_folder, train=train, download=True)
        self.images_folder = images_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对于 CIFAR10 数据集
        ])
        self.is_train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = int(self.targets[idx])
        if self.is_train:   # 如果是testdata，全部data添加白点，测试攻击成功率。
            modified_idx = random.sample(range(len(self.data)), k=int(len(self.data) * 0.1))  # 随机选择10%的数据修改
        else:
            modified_idx = random.sample(range(len(self.data)), k=int(len(self.data) * 1))  # 选择100%的数据修改

        image = Image.fromarray(image, mode='RGB') 

        if idx in modified_idx:
            label = 2
            # 在右下角填充一个 2x2 的白色区域
            # image = Image.fromarray(image.numpy(), mode='L') 
            draw =  ImageDraw.Draw(image)
            draw.rectangle([image.width - 1, image.height - 1, image.width, image.height], fill="white")

        if self.transform:
            image = self.transform(image)

        return image, label
    
# # 测试CustomMNIST
# modified_data_path = './data'
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
# dataset = CustomMNIST(images_folder=modified_data_path, train=True, transform=transform)
# data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
# for images, labels in data_loader:
#     print(images.size(), labels)
