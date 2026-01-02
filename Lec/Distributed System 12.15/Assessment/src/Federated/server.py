"""
联邦学习服务器模块
实现FedAvg算法进行模型聚合
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy


class TestDataset(Dataset):
    """测试数据集"""
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28)
        image = torch.from_numpy(image).float()
        label = self.labels[idx]
        return image, label


class ParameterServer:
    """参数服务器类，负责聚合客户端模型"""
    
    def __init__(self, model, test_images, test_labels, device='cpu'):
        """
        参数:
            model: 全局模型
            test_images: 测试集图像
            test_labels: 测试集标签
            device: 运行设备
        """
        self.device = device
        self.global_model = model.to(device)
        
        # 创建测试数据加载器
        test_dataset = TestDataset(test_images, test_labels)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 记录训练历史
        self.history = {
            'global_loss': [],
            'global_acc': []
        }
        
    def get_global_parameters(self):
        """获取全局模型参数"""
        return copy.deepcopy(self.global_model.state_dict())
    
    def federated_averaging(self, client_parameters, client_weights):
        """
        FedAvg算法：按样本数量加权平均各客户端的模型参数
        
        参数:
            client_parameters: 各客户端的模型参数列表
            client_weights: 各客户端的样本数量
        返回:
            聚合后的模型参数
        """
        # 计算总样本数
        total_samples = sum(client_weights)
        
        # 初始化聚合参数（先复制第一个客户端的参数结构）
        aggregated_params = copy.deepcopy(client_parameters[0])
        
        # 将所有参数初始化为0
        for key in aggregated_params:
            aggregated_params[key] = torch.zeros_like(
                aggregated_params[key], 
                dtype=torch.float32
            )
        
        # 加权求和
        for params, weight in zip(client_parameters, client_weights):
            # 计算该客户端的权重比例
            weight_ratio = weight / total_samples
            for key in aggregated_params:
                aggregated_params[key] += params[key].float() * weight_ratio
        
        return aggregated_params
    
    def aggregate(self, clients):
        """聚合所有客户端的模型参数"""
        # 收集各客户端的参数和样本数量
        client_parameters = []
        client_weights = []
        
        for client in clients:
            client_parameters.append(client.get_model_parameters())
            client_weights.append(client.num_samples)
        
        # 执行FedAvg聚合
        aggregated_params = self.federated_averaging(client_parameters, client_weights)
        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated_params)
        
    def distribute_model(self, clients):
        """将全局模型分发给所有客户端"""
        global_params = self.get_global_parameters()
        for client in clients:
            client.set_model_parameters(global_params)
            
    def evaluate_global_model(self):
        """在测试集上评估全局模型"""
        self.global_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def run_round(self, clients, local_epochs=1, verbose=True):
        """
        执行一轮联邦学习
        
        步骤:
        1. 分发全局模型给客户端
        2. 客户端本地训练
        3. 聚合客户端模型
        4. 评估全局模型
        """
        # 1. 分发模型
        self.distribute_model(clients)
        
        # 2. 客户端本地训练
        for client in clients:
            loss, acc = client.train(local_epochs=local_epochs)
            if verbose:
                print(f"  客户端 {client.client_id}: Loss={loss:.4f}, Acc={acc:.2f}%")
        
        # 3. 聚合模型
        self.aggregate(clients)
        
        # 4. 评估全局模型
        global_loss, global_acc = self.evaluate_global_model()
        
        # 记录历史
        self.history['global_loss'].append(global_loss)
        self.history['global_acc'].append(global_acc)
        
        return global_loss, global_acc
    
    def get_history(self):
        """获取训练历史"""
        return self.history
