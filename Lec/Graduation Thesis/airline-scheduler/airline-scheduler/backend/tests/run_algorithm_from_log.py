"""
从log目录读取最新的预处理数据并运行算法
这个脚本方便实验和调参
"""

import sys
import os
import glob

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithm.genetic_algorithm import GeneticAlgorithm
from algorithm.ant_colony_algorithm import AntColonyAlgorithm
from algorithm.particle_swarm_algorithm import ParticleSwarmAlgorithm

def find_latest_preprocessed_file(log_dir: str = "../src/output") -> str:
    """
    查找最新的预处理输出文件
    
    Args:
        log_dir: 日志根目录
    
    Returns:
        最新文件的路径
    """
    preprocessed_dir = os.path.join(log_dir, "preprocessed")
    pattern = os.path.join(preprocessed_dir, "algorithm_input_*.json")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"在 {preprocessed_dir} 中找不到预处理文件")
    
    # 按修改时间排序，返回最新的
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def run_single_algorithm(algorithm_name: str, 
                        flights: list, 
                        n_runways: int,
                        **kwargs) -> dict:
    """
    运行单个算法
    
    Args:
        algorithm_name: 算法名称 ('ga', 'aco', 'pso')
        flights: 航班列表
        n_runways: 跑道数
        **kwargs: 算法特定参数
    
    Returns:
        算法结果
    """
    if algorithm_name == 'ga':
        algo = GeneticAlgorithm(
            population_size=kwargs.get('population_size', 50),
            generations=kwargs.get('generations', 100),
            n_runways=n_runways
        )
    elif algorithm_name == 'aco':
        algo = AntColonyAlgorithm(
            n_ants=kwargs.get('n_ants', 30),
            n_iterations=kwargs.get('n_iterations', 100),
            n_runways=n_runways
        )
    elif algorithm_name == 'pso':
        algo = ParticleSwarmAlgorithm(
            n_particles=kwargs.get('n_particles', 30),
            n_iterations=kwargs.get('n_iterations', 100),
            n_runways=n_runways
        )
    else:
        raise ValueError(f"未知算法: {algorithm_name}")
    
    print(f"\n运行 {algorithm_name.upper()} 算法...")
    result = algo.optimize(flights)
    print(f"  完成 - 总延误: {result['penalty']:.2f} 分钟")
    
    return result

def main():
    """主函数"""
    print("=" * 70)
    print("从预处理数据运行算法")
    print("=" * 70)
    
    # 查找最新的预处理文件
    try:
        json_path = find_latest_preprocessed_file()
        print(f"\n找到预处理文件: {os.path.basename(json_path)}")
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 test_preprocessor.py 生成预处理数据")
        return
    
    # 加载数据
    print("\n加载预处理数据...")
    flights, metadata = GeneticAlgorithm.load_preprocessed_data(json_path)
    
    print(f"  机场: {metadata['airport']}")
    print(f"  跑道数: {metadata['n_runways']}")
    print(f"  安全间隔: {metadata['safety_interval']} 分钟")
    print(f"  总事件数: {metadata['total_events']}")
    
    # 算法参数配置（可以在这里调参）
    print("\n算法参数配置:")
    ga_params = {
        'population_size': 50,
        'generations': 100
    }
    aco_params = {
        'n_ants': 30,
        'n_iterations': 100
    }
    pso_params = {
        'n_particles': 30,
        'n_iterations': 100
    }
    
    print(f"  遗传算法: population_size={ga_params['population_size']}, generations={ga_params['generations']}")
    print(f"  蚁群算法: n_ants={aco_params['n_ants']}, n_iterations={aco_params['n_iterations']}")
    print(f"  粒子群算法: n_particles={pso_params['n_particles']}, n_iterations={pso_params['n_iterations']}")
    
    # 运行算法
    print("\n" + "-" * 70)
    print("开始运行算法...")
    print("-" * 70)
    
    n_runways = metadata['n_runways']
    
    ga_result = run_single_algorithm('ga', flights, n_runways, **ga_params)
    aco_result = run_single_algorithm('aco', flights, n_runways, **aco_params)
    pso_result = run_single_algorithm('pso', flights, n_runways, **pso_params)
    
    # 结果对比
    print("\n" + "-" * 70)
    print("算法结果对比")
    print("-" * 70)
    
    results = [
        ("遗传算法 (GA)", ga_result['penalty'], ga_result),
        ("蚁群算法 (ACO)", aco_result['penalty'], aco_result),
        ("粒子群算法 (PSO)", pso_result['penalty'], pso_result)
    ]
    
    results.sort(key=lambda x: x[1])
    
    print("\n性能排名:")
    for i, (name, penalty, _) in enumerate(results, 1):
        print(f"  {i}. {name}: {penalty:.2f} 分钟")
    
    best_name, best_penalty, best_result = results[0]
    print(f"\n最优算法: {best_name}")
    print(f"最优总延误: {best_penalty:.2f} 分钟")
    
    # 展示最优调度的前几个事件
    print("\n最优调度结果示例（前5个事件）:")
    print(f"{'航班ID':<10} {'类型':<10} {'计划时间':<20} {'调度时间':<20} {'跑道':<5} {'延误(分)':<10}")
    print("-" * 85)
    
    for event in best_result['schedule'][:5]:
        flight_id = event['flight_id']
        operation = event['operation']
        planned = event['planned_time'].strftime('%Y-%m-%d %H:%M')
        scheduled = event['scheduled_time'].strftime('%Y-%m-%d %H:%M')
        runway = event.get('runway', 'N/A')
        delay = (event['scheduled_time'] - event['planned_time']).total_seconds() / 60
        
        print(f"{flight_id:<10} {operation:<10} {planned:<20} {scheduled:<20} {runway:<5} {delay:<10.2f}")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
