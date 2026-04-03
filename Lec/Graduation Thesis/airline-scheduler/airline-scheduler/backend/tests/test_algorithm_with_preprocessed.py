"""
测试算法读取预处理数据
演示如何从预处理输出文件加载数据并运行算法
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithm.genetic_algorithm import GeneticAlgorithm
from algorithm.ant_colony_algorithm import AntColonyAlgorithm
from algorithm.particle_swarm_algorithm import ParticleSwarmAlgorithm

def test_load_and_optimize():
    """测试从预处理JSON文件加载数据并运行算法"""
    
    # 指定预处理输出文件路径
    # 注意：需要先运行 test_preprocessor.py 生成这个文件
    json_path = "../src/output/algorithm_input_20260318_201331.json"
    
    if not os.path.exists(json_path):
        print(f"错误: 找不到预处理文件 {json_path}")
        print("请先运行 test_preprocessor.py 生成预处理数据")
        return
    
    print("=" * 70)
    print("测试：从预处理数据运行算法")
    print("=" * 70)
    
    # 步骤1: 加载预处理数据
    print("\n步骤1: 加载预处理数据")
    print("-" * 70)
    
    # 三个算法都有相同的静态方法，任选一个加载即可
    flights, metadata = GeneticAlgorithm.load_preprocessed_data(json_path)
    
    print(f"加载成功:")
    print(f"  机场: {metadata['airport']}")
    print(f"  跑道数: {metadata['n_runways']}")
    print(f"  安全间隔: {metadata['safety_interval']} 分钟")
    print(f"  总事件数: {metadata['total_events']}")
    print(f"  起飞事件: {metadata['departure_events']}")
    print(f"  降落事件: {metadata['arrival_events']}")
    print(f"\n航班数据格式:")
    print(f"  第一个航班: {flights[0]}")
    
    # 步骤2: 使用元数据中的跑道数初始化算法
    print("\n步骤2: 初始化算法")
    print("-" * 70)
    
    n_runways = metadata['n_runways']
    
    ga = GeneticAlgorithm(
        population_size=30,
        generations=20,
        n_runways=n_runways
    )
    print(f"  遗传算法已初始化 (跑道数: {n_runways})")
    
    aco = AntColonyAlgorithm(
        n_ants=20,
        n_iterations=20,
        n_runways=n_runways
    )
    print(f"  蚁群算法已初始化 (跑道数: {n_runways})")
    
    pso = ParticleSwarmAlgorithm(
        n_particles=30,
        n_iterations=20,
        n_runways=n_runways
    )
    print(f"  粒子群算法已初始化 (跑道数: {n_runways})")
    
    # 步骤3: 运行算法
    print("\n步骤3: 运行算法优化")
    print("-" * 70)
    
    print("\n运行遗传算法...")
    ga_result = ga.optimize(flights)
    print(f"  完成 - 总延误: {ga_result['penalty']:.2f} 分钟")
    
    print("\n运行蚁群算法...")
    aco_result = aco.optimize(flights)
    print(f"  完成 - 总延误: {aco_result['penalty']:.2f} 分钟")
    
    print("\n运行粒子群算法...")
    pso_result = pso.optimize(flights)
    print(f"  完成 - 总延误: {pso_result['penalty']:.2f} 分钟")
    
    # 步骤4: 结果对比
    print("\n步骤4: 算法结果对比")
    print("-" * 70)
    
    results = [
        ("遗传算法", ga_result['penalty']),
        ("蚁群算法", aco_result['penalty']),
        ("粒子群算法", pso_result['penalty'])
    ]
    
    results.sort(key=lambda x: x[1])
    
    print("\n算法性能排名:")
    for i, (name, penalty) in enumerate(results, 1):
        print(f"  {i}. {name}: {penalty:.2f} 分钟")
    
    print(f"\n最优算法: {results[0][0]}")
    print(f"最优总延误: {results[0][1]:.2f} 分钟")
    
    # 步骤5: 展示部分调度结果
    print("\n步骤5: 最优调度结果示例（前10个事件）")
    print("-" * 70)
    
    best_result = ga_result if results[0][0] == "遗传算法" else \
                  aco_result if results[0][0] == "蚁群算法" else pso_result
    
    best_schedule = best_result['schedule']
    
    print(f"\n{'航班ID':<10} {'类型':<10} {'计划时间':<20} {'调度时间':<20} {'跑道':<5} {'延误(分)':<10}")
    print("-" * 85)
    
    for event in best_schedule[:10]:
        flight_id = event['flight_id']
        operation = event['operation']
        planned = event['planned_time'].strftime('%Y-%m-%d %H:%M')
        scheduled = event['scheduled_time'].strftime('%Y-%m-%d %H:%M')
        runway = event.get('runway', 'N/A')
        delay = (event['scheduled_time'] - event['planned_time']).total_seconds() / 60
        
        print(f"{flight_id:<10} {operation:<10} {planned:<20} {scheduled:<20} {runway:<5} {delay:<10.2f}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    test_load_and_optimize()
