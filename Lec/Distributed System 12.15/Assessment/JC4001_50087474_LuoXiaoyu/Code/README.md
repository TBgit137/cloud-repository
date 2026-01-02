# Federated Learning vs Centralized Learning on Fashion-MNIST

This project implements and compares Centralized Learning and Federated Learning approaches for image classification on the Fashion-MNIST dataset.

## Project Structure

```
├── Centralised/                 # Centralized learning implementation
│   ├── data_loader.py           # Data loading utilities
│   ├── model.py                 # CNN model definition
│   ├── train.py                 # Training script
│   └── logs/                    # Training logs and saved models
├── Federated/                   # Federated learning implementation
│   ├── data_partition.py        # IID and Non-IID data partitioning
│   ├── model.py                 # CNN model definition
│   ├── client.py                # Federated client implementation
│   ├── server.py                # Parameter server with FedAvg
│   ├── federated_train.py       # Main training script
│   ├── plot_experiment1.py      # Plot for client number experiment
│   ├── plot_experiment2.py      # Plot for IID vs Non-IID experiment
│   ├── comparison_analysis.py   # Centralized vs Federated comparison
│   └── logs/                    # Training logs and saved models
└── dataset/                     # Fashion-MNIST dataset
    └── archive/
        ├── fashion-mnist_train.csv
        └── fashion-mnist_test.csv
```

## Requirements

Install dependencies for each module:

```bash
# For Centralized Learning
cd Centralised
pip install -r requirements.txt

# For Federated Learning
cd Federated
pip install -r requirements.txt
```

## How to Run

### 1. Centralized Learning

1. Navigate to the Centralised folder:
   ```bash
   cd Centralised
   ```

2. Modify the dataset path in `train.py` if needed:
   ```python
   TRAIN_CSV = '../dataset/archive/fashion-mnist_train.csv'
   TEST_CSV = '../dataset/archive/fashion-mnist_test.csv'
   ```

3. Run the training script:
   ```bash
   python train.py
   ```

4. Training logs and best model will be saved in `logs/` folder.

### 2. Federated Learning

1. Navigate to the Federated folder:
   ```bash
   cd Federated
   ```

2. Modify the dataset path in `federated_train.py` if needed:
   ```python
   TRAIN_CSV = '../dataset/archive/fashion-mnist_train.csv'
   TEST_CSV = '../dataset/archive/fashion-mnist_test.csv'
   ```

3. Configure experiment parameters in `federated_train.py`:
   ```python
   NUM_CLIENTS = 10
   PARTITION_TYPE = 'iid'  # 'iid' or 'non_iid'
   NUM_ROUNDS = 200
   LOCAL_EPOCHS = 2
   ```

4. Run the training script:
   ```bash
   python federated_train.py
   ```

5. Training logs will be saved in `logs/` folder.

### 3. Generate Plots

After training, update the log file paths in the plotting scripts:

1. **Experiment 1** (Effect of client numbers):
   - Edit `plot_experiment1.py` and update the history file paths
   - Run: `python plot_experiment1.py`

2. **Experiment 2** (IID vs Non-IID):
   - Edit `plot_experiment2.py` and update the history file paths
   - Run: `python plot_experiment2.py`

3. **Centralized vs Federated Comparison**:
   - Edit `comparison_analysis.py` and update the history file paths
   - Run: `python comparison_analysis.py`

## Training Process Overview

### Centralized Learning
```
train.py
    ├── data_loader.py  → Load and preprocess Fashion-MNIST data
    └── model.py        → SimpleCNN model for training
```
The centralized approach trains a single model on the entire dataset. Data is loaded via `data_loader.py`, and the CNN model defined in `model.py` is trained directly on all available data.

### Federated Learning
```
federated_train.py
    ├── data_partition.py → Partition data among clients (IID/Non-IID)
    ├── model.py          → SimpleCNN model definition
    ├── client.py         → Local training on each client
    └── server.py         → FedAvg aggregation and global evaluation
```
The federated approach simulates distributed learning:
1. `data_partition.py` splits the training data among multiple clients
2. `server.py` distributes the global model to all clients
3. `client.py` performs local training on each client's data
4. `server.py` aggregates client models using FedAvg algorithm
5. Steps 2-4 repeat for multiple communication rounds