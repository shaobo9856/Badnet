# -*- coding: utf-8 -*-
from torchvision import datasets
import os
import random
from PIL import Image, ImageDraw

def download_mnist_data(data_path):
    # 定义原始数据集的保存路径
    original_data_path = './data/original'
    os.makedirs(original_data_path, exist_ok=True)

    # 下载 MNIST 数据集并保存到指定路径
    mnist_train = datasets.MNIST(root=original_data_path, train=True, download=True)
    mnist_test = datasets.MNIST(root=original_data_path, train=False, download=True)


if __name__ == "__main__":
    download_mnist_data('data')
