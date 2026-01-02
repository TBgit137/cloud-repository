"""
绘制不同客户端数量下的准确率和损失曲线
"""
import json
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_history(filepath):
    """加载训练历史"""
    with open(filepath, 'r') as f:
        return json.load(f)


# 加载三组实验数据
history_5 = load_history('logs/FL_iid_5clients_20251213_214202/history.json')
history_10 = load_history('logs/FL_iid_10clients_20251213_230135/history.json')
history_20 = load_history('logs/FL_iid_20clients_20251214_002255/history.json')

# 通信回合数
rounds = list(range(1, 201))

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 颜色
colors = {'5': '#1f77b4', '10': '#ff7f0e', '20': '#2ca02c'}

# 图1：准确率曲线
axes[0].plot(rounds, history_5['global_acc'], color=colors['5'], linewidth=1.5, label='5 Clients')
axes[0].plot(rounds, history_10['global_acc'], color=colors['10'], linewidth=1.5, label='10 Clients')
axes[0].plot(rounds, history_20['global_acc'], color=colors['20'], linewidth=1.5, label='20 Clients')

axes[0].set_xlabel('Communication Round', fontsize=12)
axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[0].set_title('Experiment 1: Effect of Number of Clients (IID)\nAccuracy vs Communication Rounds', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, 200)
axes[0].set_ylim(75, 95)

# 图2：损失曲线
axes[1].plot(rounds, history_5['global_loss'], color=colors['5'], linewidth=1.5, label='5 Clients')
axes[1].plot(rounds, history_10['global_loss'], color=colors['10'], linewidth=1.5, label='10 Clients')
axes[1].plot(rounds, history_20['global_loss'], color=colors['20'], linewidth=1.5, label='20 Clients')

axes[1].set_xlabel('Communication Round', fontsize=12)
axes[1].set_ylabel('Test Loss', fontsize=12)
axes[1].set_title('Experiment 1: Effect of Number of Clients (IID)\nLoss vs Communication Rounds', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, 200)

# 保存图表
plt.tight_layout()
plt.savefig('experiment1_client_numbers.png', dpi=150)
print('图表已保存: experiment1_client_numbers.png')

plt.show()

# 打印结果汇总
print('\n' + '='*60)
print('实验1结果汇总')
print('='*60)

for name, history in [('5 Clients', history_5), ('10 Clients', history_10), ('20 Clients', history_20)]:
    max_acc = max(history['global_acc'])
    final_acc = history['global_acc'][-1]
    print(f'{name}: 最高准确率={max_acc:.2f}%, 最终准确率={final_acc:.2f}%')
