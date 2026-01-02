import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class FashionMNISTDataset(Dataset):
    """Fashion-MNIST数据集加载器"""
    
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path: CSV文件路径
            transform: 数据转换操作
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
        # 分离标签和特征
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.astype(np.float32)
        
        # 归一化到[0, 1]之间
        self.images = self.images / 255.0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28)
        label = self.labels[idx]
        
        # 转换为张量
        image = torch.from_numpy(image).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(train_csv_path, test_csv_path, batch_size=32, num_workers=0):
    """
    获取训练和测试数据加载器
    
    Args:
        train_csv_path: 训练集CSV路径
        test_csv_path: 测试集CSV路径
        batch_size: 批大小
        num_workers: 数据加载线程数
        
    Returns:
        train_loader, test_loader
    """
    # 创建数据集
    train_dataset = FashionMNISTDataset(train_csv_path, transform=None)
    test_dataset = FashionMNISTDataset(test_csv_path, transform=None)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader
