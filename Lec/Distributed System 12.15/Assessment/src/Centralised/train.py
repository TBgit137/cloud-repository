"""
集中式学习训练脚本
作为联邦学习的基准对比
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
from datetime import datetime
from data_loader import get_data_loaders
from model import SimpleCNN


class CentralizedTrainer:
    """集中式模型训练器"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001, model_name='SimpleCNN'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model_name = model_name
        
        # 创建日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f'logs/{model_name}_{timestamp}'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, test_loader, epochs=20):
        """完整训练流程"""
        print(f'开始训练 {self.model_name}...')
        print(f'设备: {self.device}')
        
        start_time = time.time()
        best_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch + 1)
            test_loss, test_acc = self.evaluate(test_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            print(f'Epoch [{epoch + 1}/{epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_model(f'{self.log_dir}/best_model.pth')
                print(f'  保存最佳模型 (准确率: {test_acc:.2f}%)')
        
        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time
        self.history['training_time'] = training_time
        
        # 保存历史
        self.save_history()
        
        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        
        print(f'\n训练完成!')
        print(f'最佳测试准确率: {best_acc:.2f}%')
        print(f'总训练用时: {minutes}分{seconds}秒')
        
        return self.history
    
    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
    
    def save_history(self):
        """保存训练历史"""
        history_path = f'{self.log_dir}/history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)


def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 配置
    TRAIN_CSV = '../dataset/archive/fashion-mnist_train.csv'
    TEST_CSV = '../dataset/archive/fashion-mnist_test.csv'
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.001
    
    # 使用CPU运行
    device = torch.device('cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    print('加载数据...')
    train_loader, test_loader = get_data_loaders(TRAIN_CSV, TEST_CSV, batch_size=BATCH_SIZE)
    print(f'训练集大小: {len(train_loader.dataset)}')
    print(f'测试集大小: {len(test_loader.dataset)}')
    
    # 创建模型
    model = SimpleCNN(num_classes=10)
    
    # 训练
    trainer = CentralizedTrainer(
        model, 
        device=device, 
        learning_rate=LEARNING_RATE,
        model_name='SimpleCNN'
    )
    
    trainer.train(train_loader, test_loader, epochs=EPOCHS)


if __name__ == '__main__':
    main()
