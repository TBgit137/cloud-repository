"""
Data Partition Module
Implements IID and Non-IID data partitioning methods
"""
import pandas as pd
import numpy as np


def load_fashion_mnist(train_csv_path, test_csv_path):
    """
    Load Fashion-MNIST dataset
    Returns: train_images, train_labels, test_images, test_labels
    """
    # Read CSV files
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    
    # First column is label, remaining columns are pixel values
    train_labels = train_data.iloc[:, 0].values
    train_images = train_data.iloc[:, 1:].values.astype(np.float32) / 255.0
    
    test_labels = test_data.iloc[:, 0].values
    test_images = test_data.iloc[:, 1:].values.astype(np.float32) / 255.0
    
    return train_images, train_labels, test_images, test_labels


def partition_iid(images, labels, num_clients):
    """
    IID data partitioning (Independent and Identically Distributed)
    Randomly and uniformly distribute data to each client
    
    Args:
        images: Image data
        labels: Label data
        num_clients: Number of clients
    Returns:
        Dictionary, key is client ID, value is client's data
    """
    num_samples = len(images)
    
    # Randomly shuffle indices
    indices = np.random.permutation(num_samples)
    
    # Calculate samples per client
    samples_per_client = num_samples // num_clients
    
    client_data = {}
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            # Last client gets all remaining data
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
    Non-IID data partitioning (Non-Independent and Identically Distributed)
    Each client only receives data from partial classes
    
    Args:
        images: Image data
        labels: Label data
        num_clients: Number of clients
        shards_per_client: Number of data shards per client
    Returns:
        Dictionary, key is client ID, value is client's data
    """
    num_samples = len(images)
    
    # Sort by labels so same class data are grouped together
    sorted_indices = np.argsort(labels)
    
    # Calculate total number of shards needed
    total_shards = num_clients * shards_per_client
    shard_size = num_samples // total_shards
    
    # Create shards
    shards = []
    for i in range(total_shards):
        start_idx = i * shard_size
        if i == total_shards - 1:
            end_idx = num_samples
        else:
            end_idx = start_idx + shard_size
        shards.append(sorted_indices[start_idx:end_idx])
    
    # Randomly assign shards to clients
    shard_indices = np.random.permutation(total_shards)
    
    client_data = {}
    for i in range(num_clients):
        # Get shards for this client
        client_shards = shard_indices[i * shards_per_client:(i + 1) * shards_per_client]
        client_indices = np.concatenate([shards[s] for s in client_shards])
        
        client_data[i] = {
            'images': images[client_indices],
            'labels': labels[client_indices]
        }
    
    return client_data


def print_data_distribution(client_data, num_classes=10):
    """Print data distribution for each client"""
    print("\n" + "="*60)
    print("Client Data Distribution")
    print("="*60)
    
    for client_id, data in client_data.items():
        labels = data['labels']
        total = len(labels)
        
        # Count samples for each class
        class_counts = [np.sum(labels == c) for c in range(num_classes)]
        
        print(f"\nClient {client_id}: Total samples = {total}")
        print(f"  Class distribution: {class_counts}")
    
    print("\n" + "="*60)
