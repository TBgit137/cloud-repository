"""
Plot accuracy and loss comparison for IID and Non-IID data distribution
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


# Load IID and Non-IID experiment data
# Note: Please modify according to actual log file paths
history_iid = load_history('logs/FL_iid_10clients_20251213_230135/history.json')
history_noniid = load_history('logs/FL_non_iid_10clients_20251214_021446/history.json')

# Communication rounds
num_rounds = len(history_iid['global_acc'])
rounds = list(range(1, num_rounds + 1))

# Create charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Colors
colors = {'iid': '#1f77b4', 'non_iid': '#ff7f0e'}

# Chart 1: Accuracy curve
axes[0].plot(rounds, history_iid['global_acc'], color=colors['iid'], linewidth=1.5, label='IID')
axes[0].plot(rounds, history_noniid['global_acc'], color=colors['non_iid'], linewidth=1.5, label='Non-IID')

axes[0].set_xlabel('Communication Round', fontsize=12)
axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[0].set_title('Experiment 2: IID vs Non-IID Data Distribution\nAccuracy vs Communication Rounds', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, num_rounds)
axes[0].set_ylim(30, 100)

# Chart 2: Loss curve
axes[1].plot(rounds, history_iid['global_loss'], color=colors['iid'], linewidth=1.5, label='IID')
axes[1].plot(rounds, history_noniid['global_loss'], color=colors['non_iid'], linewidth=1.5, label='Non-IID')

axes[1].set_xlabel('Communication Round', fontsize=12)
axes[1].set_ylabel('Test Loss', fontsize=12)
axes[1].set_title('Experiment 2: IID vs Non-IID Data Distribution\nLoss vs Communication Rounds', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, num_rounds)

# Save chart
plt.tight_layout()
plt.savefig('experiment2_iid_vs_noniid.png', dpi=150)
print('Chart saved: experiment2_iid_vs_noniid.png')

plt.show()

# Print results summary
print('\n' + '='*60)
print('Experiment 2 Results Summary')
print('='*60)

iid_max = max(history_iid['global_acc'])
noniid_max = max(history_noniid['global_acc'])

print(f'IID: Max Accuracy={iid_max:.2f}%')
print(f'Non-IID: Max Accuracy={noniid_max:.2f}%')
print(f'Difference: {iid_max - noniid_max:.2f}%')
