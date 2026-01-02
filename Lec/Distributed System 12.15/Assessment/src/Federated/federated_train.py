"""
联邦学习主训练脚本
"""
import torch
import numpy as np
import json
import os
import time
from datetime import datetime

from data_partition import load_fashion_mnist, partition_iid, partition_non_iid, print_data_distribution
from client import FederatedClient
from server import ParameterServer
from model import SimpleCNN


class FederatedLearning:
    """联邦学习训练器"""
    
    def __init__(self, num_clients=10, partition_type='iid', num_rounds=50,
                 local_epochs=1, learning_rate=0.01, batch_size=32,
                 client_fraction=1.0, device='cpu'):
        """
        参数:
            num_clients: 客户端数量
            partition_type: 数据划分方式 ('iid' 或 'non_iid')
            num_rounds: 通信回合数
            local_epochs: 本地训练轮数
            learning_rate: 学习率
            batch_size: 批大小
            client_fraction: 每轮参与的客户端比例
            device: 运行设备
        """
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.client_fraction = client_fraction
        self.device = device
        
        # 创建日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f'logs/FL_{partition_type}_{num_clients}clients_{timestamp}'
        os.makedirs(self.log_dir, exist_ok=True)
        
    def setup(self, train_csv_path, test_csv_path):
        """设置联邦学习环境"""
        print("="*60)
        print("联邦学习设置")
        print("="*60)
        
        # 1. 加载数据
        print("\n[1/4] 加载Fashion-MNIST数据集...")
        train_images, train_labels, test_images, test_labels = load_fashion_mnist(
            train_csv_path, test_csv_path
        )
        print(f"  训练集: {len(train_images)} 样本")
        print(f"  测试集: {len(test_images)} 样本")
        
        # 2. 数据划分
        print(f"\n[2/4] 数据划分 (方式: {self.partition_type})...")
        if self.partition_type == 'iid':
            client_data = partition_iid(train_images, train_labels, self.num_clients)
        else:
            client_data = partition_non_iid(train_images, train_labels, self.num_clients)
        
        print_data_distribution(client_data)
        
        # 3. 创建客户端
        print(f"\n[3/4] 创建 {self.num_clients} 个客户端...")
        self.clients = []
        
        for client_id in range(self.num_clients):
            model = SimpleCNN(num_classes=10)
            client = FederatedClient(
                client_id=client_id,
                data=client_data[client_id],
                model=model,
                device=self.device,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size
            )
            self.clients.append(client)
            print(f"  {client}")
        
        # 4. 创建服务器
        print(f"\n[4/4] 创建参数服务器...")
        global_model = SimpleCNN(num_classes=10)
        self.server = ParameterServer(
            model=global_model,
            test_images=test_images,
            test_labels=test_labels,
            device=self.device
        )
        
        print("\n" + "="*60)
        print("设置完成!")
        print("="*60)
        
    def train(self, verbose=True):
        """执行联邦学习训练"""
        print("\n" + "="*60)
        print("开始联邦学习训练")
        print("="*60)
        print(f"通信回合数: {self.num_rounds}")
        print(f"本地训练轮数: {self.local_epochs}")
        print("="*60 + "\n")
        
        start_time = time.time()
        best_acc = 0
        
        for round_num in range(self.num_rounds):
            print(f"\n--- 通信回合 {round_num + 1}/{self.num_rounds} ---")
            
            # 选择参与本轮的客户端
            num_selected = max(1, int(self.num_clients * self.client_fraction))
            selected_indices = np.random.choice(
                self.num_clients, num_selected, replace=False
            )
            selected_clients = [self.clients[i] for i in selected_indices]
            
            # 执行一轮联邦学习
            global_loss, global_acc = self.server.run_round(
                clients=selected_clients,
                local_epochs=self.local_epochs,
                verbose=verbose
            )
            
            print(f"\n全局模型 - Loss: {global_loss:.4f}, Acc: {global_acc:.2f}%")
            
            # 保存最佳模型
            if global_acc > best_acc:
                best_acc = global_acc
                self.save_model(f'{self.log_dir}/best_model.pth')
                print(f"保存最佳模型 (准确率: {global_acc:.2f}%)")
        
        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time
        self.training_time = training_time
        
        # 保存训练历史
        self.save_history()
        
        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        
        print("\n" + "="*60)
        print(f"训练完成!")
        print(f"最佳测试准确率: {best_acc:.2f}%")
        print(f"总训练用时: {minutes}分{seconds}秒")
        print("="*60)
        
        return self.server.get_history()
    
    def save_model(self, path):
        """保存模型"""
        torch.save(self.server.global_model.state_dict(), path)
        
    def save_history(self):
        """保存训练历史"""
        history = self.server.get_history()
        
        save_data = {
            'global_loss': history['global_loss'],
            'global_acc': history['global_acc'],
            'training_time': self.training_time,
            'config': {
                'num_clients': self.num_clients,
                'partition_type': self.partition_type,
                'num_rounds': self.num_rounds,
                'local_epochs': self.local_epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'client_fraction': self.client_fraction
            }
        }
        
        history_path = f'{self.log_dir}/history.json'
        with open(history_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"训练历史已保存到: {history_path}")


def main():
    # 设置随机种子，确保实验可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据路径
    TRAIN_CSV = '../dataset/archive/fashion-mnist_train.csv'
    TEST_CSV = '../dataset/archive/fashion-mnist_test.csv'
    
    # 实验参数
    NUM_CLIENTS = 10
    PARTITION_TYPE = 'non_iid'  # 'iid' 或 'non_iid'
    NUM_ROUNDS = 200
    LOCAL_EPOCHS = 2
    LEARNING_RATE = 0.0008
    BATCH_SIZE = 32
    
    # 使用CPU运行（兼容性更好）
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 创建联邦学习实例
    fl = FederatedLearning(
        num_clients=NUM_CLIENTS,
        partition_type=PARTITION_TYPE,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        client_fraction=1.0,
        device=device
    )
    
    # 设置环境
    fl.setup(TRAIN_CSV, TEST_CSV)
    
    # 开始训练
    fl.train(verbose=True)


if __name__ == '__main__':
    main()
