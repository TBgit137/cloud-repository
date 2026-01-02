"""
Federated Learning Server Module
Implements FedAvg algorithm for model aggregation
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy


class TestDataset(Dataset):
    """Test dataset"""
    
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
    """Parameter Server class, responsible for aggregating client models"""
    
    def __init__(self, model, test_images, test_labels, device='cpu'):
        """
        Args:
            model: Global model
            test_images: Test set images
            test_labels: Test set labels
            device: Running device
        """
        self.device = device
        self.global_model = model.to(device)
        
        # Create test data loader
        test_dataset = TestDataset(test_images, test_labels)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Record training history
        self.history = {
            'global_loss': [],
            'global_acc': []
        }
        
    def get_global_parameters(self):
        """Get global model parameters"""
        return copy.deepcopy(self.global_model.state_dict())
    
    def federated_averaging(self, client_parameters, client_weights):
        """
        FedAvg algorithm: Weighted average of client model parameters by sample count
        
        Args:
            client_parameters: List of model parameters from each client
            client_weights: Sample count from each client
        Returns:
            Aggregated model parameters
        """
        # Calculate total samples
        total_samples = sum(client_weights)
        
        # Initialize aggregated parameters (copy structure from first client)
        aggregated_params = copy.deepcopy(client_parameters[0])
        
        # Initialize all parameters to 0
        for key in aggregated_params:
            aggregated_params[key] = torch.zeros_like(
                aggregated_params[key], 
                dtype=torch.float32
            )
        
        # Weighted sum
        for params, weight in zip(client_parameters, client_weights):
            # Calculate weight ratio for this client
            weight_ratio = weight / total_samples
            for key in aggregated_params:
                aggregated_params[key] += params[key].float() * weight_ratio
        
        return aggregated_params
    
    def aggregate(self, clients):
        """Aggregate model parameters from all clients"""
        # Collect parameters and sample counts from each client
        client_parameters = []
        client_weights = []
        
        for client in clients:
            client_parameters.append(client.get_model_parameters())
            client_weights.append(client.num_samples)
        
        # Execute FedAvg aggregation
        aggregated_params = self.federated_averaging(client_parameters, client_weights)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_params)
        
    def distribute_model(self, clients):
        """Distribute global model to all clients"""
        global_params = self.get_global_parameters()
        for client in clients:
            client.set_model_parameters(global_params)
            
    def evaluate_global_model(self):
        """Evaluate global model on test set"""
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
        Execute one round of federated learning
        
        Steps:
        1. Distribute global model to clients
        2. Client local training
        3. Aggregate client models
        4. Evaluate global model
        """
        # 1. Distribute model
        self.distribute_model(clients)
        
        # 2. Client local training
        for client in clients:
            loss, acc = client.train(local_epochs=local_epochs)
            if verbose:
                print(f"  Client {client.client_id}: Loss={loss:.4f}, Acc={acc:.2f}%")
        
        # 3. Aggregate models
        self.aggregate(clients)
        
        # 4. Evaluate global model
        global_loss, global_acc = self.evaluate_global_model()
        
        # Record history
        self.history['global_loss'].append(global_loss)
        self.history['global_acc'].append(global_acc)
        
        return global_loss, global_acc
    
    def get_history(self):
        """Get training history"""
        return self.history
