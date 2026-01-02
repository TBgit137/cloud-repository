"""
联邦学习 vs 集中式学习 性能对比
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
# 注意：请根据实际的日志文件路径修改
centralized = load_history('../Centralised/logs/SimpleCNN_20251214_131029\history.json')
fl_iid = load_history('logs/FL_iid_10clients_20251213_230135/history.json')
fl_noniid = load_history('logs/FL_non_iid_10clients_20251214_021446/history.json')


# ==================== 计算比较指标 ====================

# 1. 最终准确率和最高准确率
cent_final_acc = centralized['test_acc'][-1]
cent_max_acc = max(centralized['test_acc'])

iid_final_acc = fl_iid['global_acc'][-1]
iid_max_acc = max(fl_iid['global_acc'])

noniid_final_acc = fl_noniid['global_acc'][-1]
noniid_max_acc = max(fl_noniid['global_acc'])

# 2. 收敛轮数（达到90%准确率所需的轮数）
CONVERGENCE_THRESHOLD = 90.0

def find_convergence_round(acc_list, threshold):
    """找到首次达到阈值的轮数"""
    for i, acc in enumerate(acc_list):
        if acc >= threshold:
            return i + 1
    return None  # 未达到阈值

cent_conv = find_convergence_round(centralized['test_acc'], CONVERGENCE_THRESHOLD)
iid_conv = find_convergence_round(fl_iid['global_acc'], CONVERGENCE_THRESHOLD)
noniid_conv = find_convergence_round(fl_noniid['global_acc'], CONVERGENCE_THRESHOLD)

# 3. 训练时间
cent_time = centralized['training_time'] / 60  # 转换为分钟
iid_time = fl_iid['training_time'] / 60
noniid_time = fl_noniid['training_time'] / 60


# ==================== 打印对比结果 ====================

print('='*70)
print('联邦学习 vs 集中式学习 性能对比')
print('='*70)

print('\n【1. 最终准确率对比】')
print('-'*50)
print(f'集中式学习:        最终={cent_final_acc:.2f}%,  最高={cent_max_acc:.2f}%')
print(f'联邦学习 (IID):    最终={iid_final_acc:.2f}%,  最高={iid_max_acc:.2f}%')
print(f'联邦学习 (Non-IID): 最终={noniid_final_acc:.2f}%,  最高={noniid_max_acc:.2f}%')

print('\n【2. 收敛速度对比】(达到90%准确率所需轮数)')
print('-'*50)
if cent_conv:
    print(f'集中式学习:        {cent_conv} epochs')
else:
    print(f'集中式学习:        未达到90%')
    
if iid_conv:
    print(f'联邦学习 (IID):    {iid_conv} rounds')
else:
    print(f'联邦学习 (IID):    未达到90%')
    
if noniid_conv:
    print(f'联邦学习 (Non-IID): {noniid_conv} rounds')
else:
    print(f'联邦学习 (Non-IID): 未达到90%')

print('\n【3. 训练时间对比】')
print('-'*50)
print(f'集中式学习:        {cent_time:.1f} 分钟')
print(f'联邦学习 (IID):    {iid_time:.1f} 分钟')
print(f'联邦学习 (Non-IID): {noniid_time:.1f} 分钟')

print('\n【4. 结论】')
print('-'*50)
print(f'集中式 vs FL(IID) 准确率差异: {cent_max_acc - iid_max_acc:.2f}%')
print(f'FL(IID) vs FL(Non-IID) 准确率差异: {iid_max_acc - noniid_max_acc:.2f}%')
print('='*70)


# ==================== 绘制对比图表 ====================

# 准备数据
epochs_cent = list(range(1, len(centralized['test_acc']) + 1))
rounds_iid = list(range(1, len(fl_iid['global_acc']) + 1))
rounds_noniid = list(range(1, len(fl_noniid['global_acc']) + 1))

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 颜色
colors = {'cent': '#2ca02c', 'iid': '#1f77b4', 'noniid': '#ff7f0e'}

# 图1：准确率对比
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

# 图2：损失对比
axes[1].plot(epochs_cent, centralized['test_loss'], color=colors['cent'], linewidth=2, label='Centralized')
axes[1].plot(rounds_iid, fl_iid['global_loss'], color=colors['iid'], linewidth=1.5, label='FL (IID)')
axes[1].plot(rounds_noniid, fl_noniid['global_loss'], color=colors['noniid'], linewidth=1.5, label='FL (Non-IID)')

axes[1].set_xlabel('Epoch / Communication Round', fontsize=12)
axes[1].set_ylabel('Test Loss', fontsize=12)
axes[1].set_title('Centralized vs Federated Learning\nLoss Comparison', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, max(len(epochs_cent), len(rounds_iid)))

# 保存图表
plt.tight_layout()
plt.savefig('comparison_centralized_vs_federated.png', dpi=150)
print('\n图表已保存: comparison_centralized_vs_federated.png')

plt.show()
