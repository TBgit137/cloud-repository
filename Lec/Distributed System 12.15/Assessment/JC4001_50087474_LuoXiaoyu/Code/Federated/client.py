"""
Federated Learning Client Module
Simulates devices participating in federated learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy


class ClientDataset(Dataset):
    """Client local dataset"""
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Reshape image to 28x28
        image = self.images[idx].reshape(28, 28)
        image = torch.from_numpy(image).float()
        label = self.labels[idx]
        return image, label


class FederatedClient:
    """Federated Learning Client Class"""
    
    def __init__(self, client_id, data, model, device='cpu',
                 learning_rate=0.01, batch_size=32):
        """
        Args:
            client_id: Client ID
            data: Local data containing 'images' and 'labels'
            model: Neural network model
            device: Running device
            learning_rate: Learning rate
            batch_size: Batch size
        """
        self.client_id = client_id
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Copy model as local model
        self.model = copy.deepcopy(model).to(device)
        
        # Create data loader
        self.dataset = ClientDataset(data['images'], data['labels'])
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Record local data size for weighted aggregation
        self.num_samples = len(self.dataset)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def set_model_parameters(self, parameters):
        """Receive model parameters from server"""
        self.model.load_state_dict(parameters)
        
    def get_model_parameters(self):
        """Get local model parameters to send to server"""
        return copy.deepcopy(self.model.state_dict())
    
    def train(self, local_epochs=1):
        """
        Train model on local data
        Returns: (average loss, accuracy)
        """
        self.model.train()
        
        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for epoch in range(local_epochs):
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate loss and accuracy
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def __repr__(self):
        return f"Client(id={self.client_id}, samples={self.num_samples})"
