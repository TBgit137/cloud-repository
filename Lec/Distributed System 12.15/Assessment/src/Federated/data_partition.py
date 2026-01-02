"""
数据划分模块
实现IID和Non-IID两种数据划分方式
"""
import pandas as pd
import numpy as np


def load_fashion_mnist(train_csv_path, test_csv_path):
    """
    加载Fashion-MNIST数据集
    返回: train_images, train_labels, test_images, test_labels
    """
    # 读取CSV文件
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    
    # 第一列是标签，其余列是像素值
    train_labels = train_data.iloc[:, 0].values
    train_images = train_data.iloc[:, 1:].values.astype(np.float32) / 255.0
    
    test_labels = test_data.iloc[:, 0].values
    test_images = test_data.iloc[:, 1:].values.astype(np.float32) / 255.0
    
    return train_images, train_labels, test_images, test_labels


def partition_iid(images, labels, num_clients):
    """
    IID数据划分（独立同分布）
    将数据随机均匀分配给各个客户端
    
    参数:
        images: 图像数据
        labels: 标签数据
        num_clients: 客户端数量
    返回:
        字典，key是客户端ID，value是该客户端的数据
    """
    num_samples = len(images)
    
    # 随机打乱索引
    indices = np.random.permutation(num_samples)
    
    # 计算每个客户端分配多少数据
    samples_per_client = num_samples // num_clients
    
    client_data = {}
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            # 最后一个客户端获取剩余所有数据
            end_idx = num_samples
        else:
            end_idx = start_idx + samples_per_client
        
        client_indices = indices[start_idx:end_idx]
        client_data[i] = {
            'images': images[client_indices],
            'labels': labels[client_indices]
        }
    
    return client_data


def partition_non_iid(images, labels, num_clients, shards_per_client=2):
    """
    Non-IID数据划分（非独立同分布）
    每个客户端只获得部分类别的数据
    
    参数:
        images: 图像数据
        labels: 标签数据
        num_clients: 客户端数量
        shards_per_client: 每个客户端获得的数据分片数
    返回:
        字典，key是客户端ID，value是该客户端的数据
    """
    num_samples = len(images)
    
    # 按标签排序，这样相同类别的数据会聚在一起
    sorted_indices = np.argsort(labels)
    
    # 计算总共需要多少个分片
    total_shards = num_clients * shards_per_client
    shard_size = num_samples // total_shards
    
    # 创建分片
    shards = []
    for i in range(total_shards):
        start_idx = i * shard_size
        if i == total_shards - 1:
            end_idx = num_samples
        else:
            end_idx = start_idx + shard_size
        shards.append(sorted_indices[start_idx:end_idx])
    
    # 随机分配分片给客户端
    shard_indices = np.random.permutation(total_shards)
    
    client_data = {}
    for i in range(num_clients):
        # 获取该客户端的分片
        client_shards = shard_indices[i * shards_per_client:(i + 1) * shards_per_client]
        client_indices = np.concatenate([shards[s] for s in client_shards])
        
        client_data[i] = {
            'images': images[client_indices],
            'labels': labels[client_indices]
        }
    
    return client_data


def print_data_distribution(client_data, num_classes=10):
    """打印各客户端的数据分布情况"""
    print("\n" + "="*60)
    print("客户端数据分布")
    print("="*60)
    
    for client_id, data in client_data.items():
        labels = data['labels']
        total = len(labels)
        
        # 统计每个类别的数量
        class_counts = [np.sum(labels == c) for c in range(num_classes)]
        
        print(f"\n客户端 {client_id}: 总样本数 = {total}")
        print(f"  类别分布: {class_counts}")
    
    print("\n" + "="*60)
