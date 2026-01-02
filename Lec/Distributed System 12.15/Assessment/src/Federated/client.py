"""
联邦学习客户端模块
模拟参与联邦学习的设备
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy


class ClientDataset(Dataset):
    """客户端本地数据集"""
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 将图像reshape成28x28
        image = self.images[idx].reshape(28, 28)
        image = torch.from_numpy(image).float()
        label = self.labels[idx]
        return image, label


class FederatedClient:
    """联邦学习客户端类"""
    
    def __init__(self, client_id, data, model, device='cpu',
                 learning_rate=0.01, batch_size=32):
        """
        参数:
            client_id: 客户端ID
            data: 本地数据，包含'images'和'labels'
            model: 神经网络模型
            device: 运行设备
            learning_rate: 学习率
            batch_size: 批大小
        """
        self.client_id = client_id
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # 复制一份模型作为本地模型
        self.model = copy.deepcopy(model).to(device)
        
        # 创建数据加载器
        self.dataset = ClientDataset(data['images'], data['labels'])
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 记录本地数据量，用于后续加权聚合
        self.num_samples = len(self.dataset)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
    def set_model_parameters(self, parameters):
        """从服务器接收模型参数"""
        self.model.load_state_dict(parameters)
        
    def get_model_parameters(self):
        """获取本地模型参数，发送给服务器"""
        return copy.deepcopy(self.model.state_dict())
    
    def train(self, local_epochs=1):
        """
        在本地数据上训练模型
        返回: (平均损失, 准确率)
        """
        self.model.train()
        
        # 创建优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for epoch in range(local_epochs):
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计损失和准确率
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def __repr__(self):
        return f"Client(id={self.client_id}, samples={self.num_samples})"
