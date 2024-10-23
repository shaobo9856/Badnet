import pandas as pd
import torch
from torch.utils.data import DataLoader
from minst import CustomMNIST,CustomCIFAR10
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import argparse


class BadNet(nn.Module):
    def __init__(self,in_channels):
        super(BadNet, self).__init__()
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5, 5), stride=1, padding=0)  # 1x28x28 -> 16x24x24
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x24x24 -> 16x12x12
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=1, padding=0)  # 16x12x12 -> 32x8x8
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 32x8x8 -> 32x4x4
        
        # 定义全连接层
        features = 800 if in_channels == 3 else 32 * 4 * 4
        self.fc1 = nn.Linear(features, 512)  # 32x4x4 -> 512
        self.fc2 = nn.Linear(512, 10)  # 512 -> 10 (分类)

    def forward(self, x):
        # 前向传播
        x = self.pool1(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool1
        x = self.pool2(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool2
        
        # x = x.view(-1, 32 * 4 * 4) 
        x = x.view(x.size(0), -1)  # 展平，使用 batch size
        x = F.relu(self.fc1(x))  # FC1 + ReLU
        x = self.fc2(x)   
        
        return F.log_softmax(x, dim=1)  # Softmax 输出


        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default='mnist', help='dataset: mnist, cifar10')
    opt = parser.parse_args()
    if(opt.data == "mnist"):
        channels = 1
        train_data = DataLoader(CustomMNIST(images_folder="./data",train=True), batch_size=64, shuffle=True)
        test_data = DataLoader(CustomMNIST(images_folder="./data",train=False), batch_size=64, shuffle=True)
    else:
        channels = 3
        train_data = DataLoader(CustomCIFAR10(images_folder="./data",train=True), batch_size=100, shuffle=True)
        test_data = DataLoader(CustomCIFAR10(images_folder="./data",train=False), batch_size=100, shuffle=True)

    print("begin")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # transform_mnist = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))  # 对于 MNIST 数据集
    # ])

    model = BadNet(channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("start train")
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Epoch: {epoch} data: {i} loss: {running_loss/len(train_data)}")
        print(f"Epoch {epoch+1} loss: {running_loss/len(train_data)}")

    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the Badnet on the test images: {100 * correct / total}%")


if __name__ == "__main__":
    main()
