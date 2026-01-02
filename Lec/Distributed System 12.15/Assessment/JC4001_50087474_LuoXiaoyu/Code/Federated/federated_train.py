"""
Federated Learning Main Training Script
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
    """Federated Learning Trainer"""
    
    def __init__(self, num_clients=10, partition_type='iid', num_rounds=50,
                 local_epochs=1, learning_rate=0.01, batch_size=32,
                 client_fraction=1.0, device='cpu'):
        """
        Args:
            num_clients: Number of clients
            partition_type: Data partition method ('iid' or 'non_iid')
            num_rounds: Number of communication rounds
            local_epochs: Number of local training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            client_fraction: Fraction of clients participating per round
            device: Running device
        """
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.client_fraction = client_fraction
        self.device = device
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f'logs/FL_{partition_type}_{num_clients}clients_{timestamp}'
        os.makedirs(self.log_dir, exist_ok=True)
        
    def setup(self, train_csv_path, test_csv_path):
        """Setup federated learning environment"""
        print("="*60)
        print("Federated Learning Setup")
        print("="*60)
        
        # 1. Load data
        print("\n[1/4] Loading Fashion-MNIST dataset...")
        train_images, train_labels, test_images, test_labels = load_fashion_mnist(
            train_csv_path, test_csv_path
        )
        print(f"  Training set: {len(train_images)} samples")
        print(f"  Test set: {len(test_images)} samples")
        
        # 2. Data partition
        print(f"\n[2/4] Data partitioning (method: {self.partition_type})...")
        if self.partition_type == 'iid':
            client_data = partition_iid(train_images, train_labels, self.num_clients)
        else:
            client_data = partition_non_iid(train_images, train_labels, self.num_clients)
        
        print_data_distribution(client_data)
        
        # 3. Create clients
        print(f"\n[3/4] Creating {self.num_clients} clients...")
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
        
        # 4. Create server
        print(f"\n[4/4] Creating parameter server...")
        global_model = SimpleCNN(num_classes=10)
        self.server = ParameterServer(
            model=global_model,
            test_images=test_images,
            test_labels=test_labels,
            device=self.device
        )
        
        print("\n" + "="*60)
        print("Setup completed!")
        print("="*60)
        
    def train(self, verbose=True):
        """Execute federated learning training"""
        print("\n" + "="*60)
        print("Starting Federated Learning Training")
        print("="*60)
        print(f"Communication rounds: {self.num_rounds}")
        print(f"Local training epochs: {self.local_epochs}")
        print("="*60 + "\n")
        
        start_time = time.time()
        best_acc = 0
        
        for round_num in range(self.num_rounds):
            print(f"\n--- Communication Round {round_num + 1}/{self.num_rounds} ---")
            
            # Select clients participating in this round
            num_selected = max(1, int(self.num_clients * self.client_fraction))
            selected_indices = np.random.choice(
                self.num_clients, num_selected, replace=False
            )
            selected_clients = [self.clients[i] for i in selected_indices]
            
            # Execute one round of federated learning
            global_loss, global_acc = self.server.run_round(
                clients=selected_clients,
                local_epochs=self.local_epochs,
                verbose=verbose
            )
            
            print(f"\nGlobal Model - Loss: {global_loss:.4f}, Acc: {global_acc:.2f}%")
            
            # Save best model
            if global_acc > best_acc:
                best_acc = global_acc
                self.save_model(f'{self.log_dir}/best_model.pth')
                print(f"Saved best model (Accuracy: {global_acc:.2f}%)")
        
        # Calculate training time
        end_time = time.time()
        training_time = end_time - start_time
        self.training_time = training_time
        
        # Save training history
        self.save_history()
        
        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        
        print("\n" + "="*60)
        print(f"Training completed!")
        print(f"Best test accuracy: {best_acc:.2f}%")
        print(f"Total training time: {minutes}m {seconds}s")
        print("="*60)
        
        return self.server.get_history()
    
    def save_model(self, path):
        """Save model"""
        torch.save(self.server.global_model.state_dict(), path)
        
    def save_history(self):
        """Save training history"""
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
        print(f"Training history saved to: {history_path}")


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths
    TRAIN_CSV = '../dataset/archive/fashion-mnist_train.csv'
    TEST_CSV = '../dataset/archive/fashion-mnist_test.csv'
    
    # Experiment parameters
    NUM_CLIENTS = 10
    PARTITION_TYPE = 'non_iid'  # 'iid' or 'non_iid'
    NUM_ROUNDS = 200
    LOCAL_EPOCHS = 2
    LEARNING_RATE = 0.0008
    BATCH_SIZE = 32
    
    # Use CPU (better compatibility)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create federated learning instance
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
    
    # Setup environment
    fl.setup(TRAIN_CSV, TEST_CSV)
    
    # Start training
    fl.train(verbose=True)


if __name__ == '__main__':
    main()
