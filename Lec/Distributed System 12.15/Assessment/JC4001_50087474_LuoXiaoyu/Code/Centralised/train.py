"""
Centralized Learning Training Script
As a baseline comparison for Federated Learning
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
    """Centralized Model Trainer"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001, model_name='SimpleCNN'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model_name = model_name
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f'logs/{model_name}_{timestamp}'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Record training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
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
        """Evaluate model"""
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
        """Complete training process"""
        print(f'Starting training {self.model_name}...')
        print(f'Device: {self.device}')
        
        start_time = time.time()
        best_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch + 1)
            test_loss, test_acc = self.evaluate(test_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            print(f'Epoch [{epoch + 1}/{epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_model(f'{self.log_dir}/best_model.pth')
                print(f'  Saved best model (Accuracy: {test_acc:.2f}%)')
        
        # Calculate training time
        end_time = time.time()
        training_time = end_time - start_time
        self.history['training_time'] = training_time
        
        # Save history
        self.save_history()
        
        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        
        print(f'\nTraining completed!')
        print(f'Best test accuracy: {best_acc:.2f}%')
        print(f'Total training time: {minutes}m {seconds}s')
        
        return self.history
    
    def save_model(self, path):
        """Save model"""
        torch.save(self.model.state_dict(), path)
    
    def save_history(self):
        """Save training history"""
        history_path = f'{self.log_dir}/history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)


def main():
    # Set random seed
    torch.manual_seed(42)
    
    # Configuration
    TRAIN_CSV = '../dataset/archive/fashion-mnist_train.csv'
    TEST_CSV = '../dataset/archive/fashion-mnist_test.csv'
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.001
    
    # Use CPU
    device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_loader, test_loader = get_data_loaders(TRAIN_CSV, TEST_CSV, batch_size=BATCH_SIZE)
    print(f'Training set size: {len(train_loader.dataset)}')
    print(f'Test set size: {len(test_loader.dataset)}')
    
    # Create model
    model = SimpleCNN(num_classes=10)
    
    # Train
    trainer = CentralizedTrainer(
        model, 
        device=device, 
        learning_rate=LEARNING_RATE,
        model_name='SimpleCNN'
    )
    
    trainer.train(train_loader, test_loader, epochs=EPOCHS)


if __name__ == '__main__':
    main()
