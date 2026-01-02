import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class FashionMNISTDataset(Dataset):
    """Fashion-MNIST Dataset Loader"""
    
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path: Path to CSV file
            transform: Data transformation operations
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
        # Separate labels and features
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.astype(np.float32)
        
        # Normalize to [0, 1]
        self.images = self.images / 255.0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28)
        label = self.labels[idx]
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(train_csv_path, test_csv_path, batch_size=32, num_workers=0):
    """
    Get training and testing data loaders
    
    Args:
        train_csv_path: Path to training CSV file
        test_csv_path: Path to testing CSV file
        batch_size: Batch size
        num_workers: Number of data loading threads
        
    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = FashionMNISTDataset(train_csv_path, transform=None)
    test_dataset = FashionMNISTDataset(test_csv_path, transform=None)
    
    # Create data loaders
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
