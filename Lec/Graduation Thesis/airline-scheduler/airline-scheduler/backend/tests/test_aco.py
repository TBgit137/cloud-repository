"""
测试蚁群算法
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithm.ant_colony_algorithm import AntColonyAlgorithm

# ==================== 配置参数 ====================
# 修改这些参数来配置算法

# 找到最新的预处理文件
import glob
log_dir = "../src/output"
pattern = os.path.join(log_dir, "preprocessed", "algorithm_input_*.json")
files = glob.glob(pattern)

if not files:
    print("错误: 找不到预处理文件")
    print("请先运行 test_preprocessor.py 生成预处理数据")
    sys.exit(1)

# 使用最新的文件
json_path = max(files, key=os.path.getmtime)
print(f"使用预处理文件: {os.path.basename(json_path)}")

# 算法参数
ALGORITHM_PARAMS = {
    'n_ants': 30,
    'n_iterations': 100,
    'alpha': 1.0,
    'beta': 2.0,
    'evaporation_rate': 0.5,
    'q': 100,
    'n_runways': None  # None表示从预处理文件中读取
}

# ==================================================

def test_aco():
    """测试蚁群算法"""
    print("=" * 70)
    print("蚁群算法测试")
    print("=" * 70)
    
    # 加载预处理数据
    print("\n加载预处理数据...")
    flights, metadata = AntColonyAlgorithm.load_preprocessed_data(json_path)
    
    print(f"  机场: {metadata['airport']}")
    print(f"  跑道数: {metadata['n_runways']}")
    print(f"  安全间隔: {metadata['safety_interval']} 分钟")
    print(f"  总事件数: {metadata['total_events']}")
    print(f"  航班数量: {len(flights)}")
    
    # 如果n_runways为None，使用预处理文件中的值
    if ALGORITHM_PARAMS['n_runways'] is None:
        ALGORITHM_PARAMS['n_runways'] = metadata['n_runways']
    
    # 初始化算法
    print(f"\n初始化蚁群算法...")
    print(f"  蚂蚁数量: {ALGORITHM_PARAMS['n_ants']}")
    print(f"  迭代次数: {ALGORITHM_PARAMS['n_iterations']}")
    print(f"  信息素因子 alpha: {ALGORITHM_PARAMS['alpha']}")
    print(f"  启发式因子 beta: {ALGORITHM_PARAMS['beta']}")
    print(f"  挥发率: {ALGORITHM_PARAMS['evaporation_rate']}")
    print(f"  信息素强度 q: {ALGORITHM_PARAMS['q']}")
    print(f"  跑道数量: {ALGORITHM_PARAMS['n_runways']}")
    
    aco = AntColonyAlgorithm(
        n_ants=ALGORITHM_PARAMS['n_ants'],
        n_iterations=ALGORITHM_PARAMS['n_iterations'],
        alpha=ALGORITHM_PARAMS['alpha'],
        beta=ALGORITHM_PARAMS['beta'],
        evaporation_rate=ALGORITHM_PARAMS['evaporation_rate'],
        q=ALGORITHM_PARAMS['q'],
        n_runways=ALGORITHM_PARAMS['n_runways']
    )
    
    # 运行优化
    print("\n运行优化...")
    result = aco.optimize(flights)
    
    # 输出结果
    print("\n" + "=" * 70)
    print("优化完成！")
    print("=" * 70)
    
    print(f"\n算法: {result['algorithm']}")
    print(f"总延误: {result['penalty']:.2f} 分钟")
    print(f"迭代次数: {result['iterations']}")
    print(f"调度航班数: {len(result['schedule'])}")
    
    # 显示前10个调度结果
    print("\n前10个航班调度结果:")
    print(f"{'航班ID':<10} {'类型':<10} {'计划时间':<20} {'调度时间':<20} {'跑道':<5} {'延误(分)':<10}")
    print("-" * 85)
    
    for event in result['schedule'][:10]:
        flight_id = event['flight_id']
        operation = event['operation']
        planned = event['planned_time'].strftime('%Y-%m-%d %H:%M')
        scheduled = event['scheduled_time'].strftime('%Y-%m-%d %H:%M')
        runway = event.get('runway', 'N/A')
        delay = (event['scheduled_time'] - event['planned_time']).total_seconds() / 60
        
        print(f"{flight_id:<10} {operation:<10} {planned:<20} {scheduled:<20} {runway:<5} {delay:<10.2f}")
    
    # 显示收敛曲线
    print("\n收敛曲线（前20次迭代）:")
    print("迭代\t\t最佳惩罚值")
    print("-" * 30)
    for i, penalty in enumerate(result['penalty_history'][:20]):
        print(f"{i+1}\t\t{penalty:.2f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_aco()
