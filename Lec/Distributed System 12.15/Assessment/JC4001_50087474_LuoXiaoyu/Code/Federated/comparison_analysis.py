"""
Federated Learning vs Centralized Learning Performance Comparison
"""
import json
import matplotlib.pyplot as plt

# Set font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_history(filepath):
    """Load training history"""
    with open(filepath, 'r') as f:
        return json.load(f)


# Load three sets of experimental data
# Note: Please modify according to actual log file paths
centralized = load_history('../Centralised/logs/SimpleCNN_20251214_131029\history.json')
fl_iid = load_history('logs/FL_iid_10clients_20251213_230135/history.json')
fl_noniid = load_history('logs/FL_non_iid_10clients_20251214_021446/history.json')


# ==================== Calculate Comparison Metrics ====================

# 1. Final accuracy and maximum accuracy
cent_final_acc = centralized['test_acc'][-1]
cent_max_acc = max(centralized['test_acc'])

iid_final_acc = fl_iid['global_acc'][-1]
iid_max_acc = max(fl_iid['global_acc'])

noniid_final_acc = fl_noniid['global_acc'][-1]
noniid_max_acc = max(fl_noniid['global_acc'])

# 2. Convergence rounds (rounds needed to reach 90% accuracy)
CONVERGENCE_THRESHOLD = 90.0

def find_convergence_round(acc_list, threshold):
    """Find the first round that reaches the threshold"""
    for i, acc in enumerate(acc_list):
        if acc >= threshold:
            return i + 1
    return None  # Threshold not reached

cent_conv = find_convergence_round(centralized['test_acc'], CONVERGENCE_THRESHOLD)
iid_conv = find_convergence_round(fl_iid['global_acc'], CONVERGENCE_THRESHOLD)
noniid_conv = find_convergence_round(fl_noniid['global_acc'], CONVERGENCE_THRESHOLD)

# 3. Training time
cent_time = centralized['training_time'] / 60  # Convert to minutes
iid_time = fl_iid['training_time'] / 60
noniid_time = fl_noniid['training_time'] / 60


# ==================== Print Comparison Results ====================

print('='*70)
print('Federated Learning vs Centralized Learning Performance Comparison')
print('='*70)

print('\n[1. Final Accuracy Comparison]')
print('-'*50)
print(f'Centralized:       Final={cent_final_acc:.2f}%,  Max={cent_max_acc:.2f}%')
print(f'FL (IID):          Final={iid_final_acc:.2f}%,  Max={iid_max_acc:.2f}%')
print(f'FL (Non-IID):      Final={noniid_final_acc:.2f}%,  Max={noniid_max_acc:.2f}%')

print('\n[2. Convergence Speed Comparison] (Rounds to reach 90% accuracy)')
print('-'*50)
if cent_conv:
    print(f'Centralized:       {cent_conv} epochs')
else:
    print(f'Centralized:       Did not reach 90%')
    
if iid_conv:
    print(f'FL (IID):          {iid_conv} rounds')
else:
    print(f'FL (IID):          Did not reach 90%')
    
if noniid_conv:
    print(f'FL (Non-IID):      {noniid_conv} rounds')
else:
    print(f'FL (Non-IID):      Did not reach 90%')

print('\n[3. Training Time Comparison]')
print('-'*50)
print(f'Centralized:       {cent_time:.1f} minutes')
print(f'FL (IID):          {iid_time:.1f} minutes')
print(f'FL (Non-IID):      {noniid_time:.1f} minutes')

print('\n[4. Conclusion]')
print('-'*50)
print(f'Centralized vs FL(IID) accuracy difference: {cent_max_acc - iid_max_acc:.2f}%')
print(f'FL(IID) vs FL(Non-IID) accuracy difference: {iid_max_acc - noniid_max_acc:.2f}%')
print('='*70)


# ==================== Plot Comparison Charts ====================

# Prepare data
epochs_cent = list(range(1, len(centralized['test_acc']) + 1))
rounds_iid = list(range(1, len(fl_iid['global_acc']) + 1))
rounds_noniid = list(range(1, len(fl_noniid['global_acc']) + 1))

# Create charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Colors
colors = {'cent': '#2ca02c', 'iid': '#1f77b4', 'noniid': '#ff7f0e'}

# Chart 1: Accuracy comparison
axes[0].plot(epochs_cent, centralized['test_acc'], color=colors['cent'], linewidth=2, label='Centralized')
axes[0].plot(rounds_iid, fl_iid['global_acc'], color=colors['iid'], linewidth=1.5, label='FL (IID)')
axes[0].plot(rounds_noniid, fl_noniid['global_acc'], color=colors['noniid'], linewidth=1.5, label='FL (Non-IID)')
axes[0].axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% Threshold')

axes[0].set_xlabel('Epoch / Communication Round', fontsize=12)
axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[0].set_title('Centralized vs Federated Learning\nAccuracy Comparison', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, max(len(epochs_cent), len(rounds_iid)))
axes[0].set_ylim(30, 100)

# Chart 2: Loss comparison
axes[1].plot(epochs_cent, centralized['test_loss'], color=colors['cent'], linewidth=2, label='Centralized')
axes[1].plot(rounds_iid, fl_iid['global_loss'], color=colors['iid'], linewidth=1.5, label='FL (IID)')
axes[1].plot(rounds_noniid, fl_noniid['global_loss'], color=colors['noniid'], linewidth=1.5, label='FL (Non-IID)')

axes[1].set_xlabel('Epoch / Communication Round', fontsize=12)
axes[1].set_ylabel('Test Loss', fontsize=12)
axes[1].set_title('Centralized vs Federated Learning\nLoss Comparison', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, max(len(epochs_cent), len(rounds_iid)))

# Save chart
plt.tight_layout()
plt.savefig('comparison_centralized_vs_federated.png', dpi=150)
print('\nChart saved: comparison_centralized_vs_federated.png')

plt.show()
