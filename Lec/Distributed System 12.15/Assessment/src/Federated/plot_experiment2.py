"""
绘制IID和Non-IID数据分布下的准确率和损失对比
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


# 加载IID和Non-IID实验数据
# 注意：请根据实际的日志文件路径修改
history_iid = load_history('logs/FL_iid_10clients_20251213_230135/history.json')
history_noniid = load_history('logs/FL_non_iid_10clients_20251214_021446/history.json')

# 通信回合数
num_rounds = len(history_iid['global_acc'])
rounds = list(range(1, num_rounds + 1))

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 颜色
colors = {'iid': '#1f77b4', 'non_iid': '#ff7f0e'}

# 图1：准确率曲线
axes[0].plot(rounds, history_iid['global_acc'], color=colors['iid'], linewidth=1.5, label='IID')
axes[0].plot(rounds, history_noniid['global_acc'], color=colors['non_iid'], linewidth=1.5, label='Non-IID')

axes[0].set_xlabel('Communication Round', fontsize=12)
axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[0].set_title('Experiment 2: IID vs Non-IID Data Distribution\nAccuracy vs Communication Rounds', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, num_rounds)
axes[0].set_ylim(30, 100)

# 图2：损失曲线
axes[1].plot(rounds, history_iid['global_loss'], color=colors['iid'], linewidth=1.5, label='IID')
axes[1].plot(rounds, history_noniid['global_loss'], color=colors['non_iid'], linewidth=1.5, label='Non-IID')

axes[1].set_xlabel('Communication Round', fontsize=12)
axes[1].set_ylabel('Test Loss', fontsize=12)
axes[1].set_title('Experiment 2: IID vs Non-IID Data Distribution\nLoss vs Communication Rounds', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, num_rounds)

# 保存图表
plt.tight_layout()
plt.savefig('experiment2_iid_vs_noniid.png', dpi=150)
print('图表已保存: experiment2_iid_vs_noniid.png')

plt.show()

# 打印结果汇总
print('\n' + '='*60)
print('实验2结果汇总')
print('='*60)

iid_max = max(history_iid['global_acc'])
noniid_max = max(history_noniid['global_acc'])

print(f'IID: 最高准确率={iid_max:.2f}%')
print(f'Non-IID: 最高准确率={noniid_max:.2f}%')
print(f'差异: {iid_max - noniid_max:.2f}%')
