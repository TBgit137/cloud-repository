"""
测试遗传算法
"""

import sys
import os
import json
import csv
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithm.genetic_algorithm import GeneticAlgorithm

# ==================== 配置参数 ====================
# 修改这些参数来配置算法

# 找到最新的预处理文件
import glob
log_dir = "../src/output"
preprocessed_dir = os.path.join(log_dir, "preprocessed")
pattern = os.path.join(preprocessed_dir, "algorithm_input_*.json")
files = glob.glob(pattern)

if not files:
    print("Error: No preprocessed file found")
    print("Please run test_preprocessor.py first to generate preprocessed data")
    sys.exit(1)

# 使用最新的文件
json_path = max(files, key=os.path.getmtime)
print(f"Using preprocessed file: {os.path.basename(json_path)}")

# 算法参数（在这里调参，所有输出和算法初始化都会随之改变）
ALGORITHM_PARAMS = {
    'population_size': 100,    # 种群大小
    'generations':     100,   # 迭代代数
    'crossover_rate':  0.8,   # 交叉概率
    'mutation_rate':   0.5,   # 变异概率
    'elite_size':      2,     # 精英保留数量
    'max_offset':      5,     # 最大时间偏移量（分钟）
    'n_runways':       None   # None 表示从预处理文件中读取
}

# ==================================================

def test_ga():
    """测试遗传算法"""
    print("=" * 70)
    print("Genetic Algorithm Test")
    print("=" * 70)

    # 加载预处理数据
    print("\nLoading preprocessed data...")
    flights, metadata = GeneticAlgorithm.load_preprocessed_data(json_path)

    print(f"  Airport: {metadata['airport']}")
    print(f"  Runways: {metadata['n_runways']}")
    print(f"  Safety interval: {metadata['safety_interval']} minutes")
    print(f"  Total events: {metadata['total_events']}")
    print(f"  Number of flights: {len(flights)}")

    # 如果n_runways为None，使用预处理文件中的值
    if ALGORITHM_PARAMS['n_runways'] is None:
        ALGORITHM_PARAMS['n_runways'] = metadata['n_runways']

    # 初始化算法
    print(f"\nInitializing Genetic Algorithm...")
    print(f"  Population size:   {ALGORITHM_PARAMS['population_size']}")
    print(f"  Generations:       {ALGORITHM_PARAMS['generations']}")
    print(f"  Crossover rate:    {ALGORITHM_PARAMS['crossover_rate']}")
    print(f"  Mutation rate:     {ALGORITHM_PARAMS['mutation_rate']}")
    print(f"  Elite size:        {ALGORITHM_PARAMS['elite_size']}")
    print(f"  Max offset:        {ALGORITHM_PARAMS['max_offset']} minutes")
    print(f"  Number of runways: {ALGORITHM_PARAMS['n_runways']}")

    ga = GeneticAlgorithm(
        population_size=ALGORITHM_PARAMS['population_size'],
        generations=ALGORITHM_PARAMS['generations'],
        crossover_rate=ALGORITHM_PARAMS['crossover_rate'],
        mutation_rate=ALGORITHM_PARAMS['mutation_rate'],
        elite_size=ALGORITHM_PARAMS['elite_size'],
        max_offset=ALGORITHM_PARAMS['max_offset'],
        n_runways=ALGORITHM_PARAMS['n_runways']
    )

    # 运行优化
    print("\nRunning optimization...")
    result = ga.optimize(flights)

    # 输出结果
    print("\n" + "=" * 70)
    print("Optimization complete!")
    print("=" * 70)

    print(f"\nAlgorithm: {result['algorithm']}")
    print(f"Total penalty: {result['penalty']:.2f} minutes")
    print(f"Generations: {result['generations']}")
    print(f"Scheduled flights: {len(result['schedule'])}")

    # 显示前10个调度结果
    print("\nFirst 10 flight schedule results:")
    print(f"{'Flight ID':<10} {'Type':<10} {'Planned Time':<20} {'Scheduled Time':<20} {'Runway':<5} {'Delay(min)':<10}")
    print("-" * 85)

    for event in result['schedule'][:10]:
        flight_id = event['flight_id']
        operation = event['operation']
        planned = event['planned_time'].strftime('%Y-%m-%d %H:%M')
        scheduled = event['scheduled_time'].strftime('%Y-%m-%d %H:%M')
        runway = event.get('runway', 'N/A')
        delay = (event['scheduled_time'] - event['planned_time']).total_seconds() / 60

        print(f"{flight_id:<10} {operation:<10} {planned:<20} {scheduled:<20} {runway:<5} {delay:<10.2f}")

    # 保存并验证结果
    _save_and_validate(result, metadata, log_dir)
    print("\n" + "=" * 70)


def _save_and_validate(result: dict, metadata: dict, log_dir: str):
    """保存调度结果并输出验证统计"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    schedule = result['schedule']

    # ---- 1. 保存 CSV ----
    results_dir = os.path.join(log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"schedule_result_{timestamp}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['flight_id', 'operation', 'planned_time',
                         'scheduled_time', 'runway', 'delay_minutes'])
        for ev in schedule:
            delay = (ev['scheduled_time'] - ev['planned_time']).total_seconds() / 60
            writer.writerow([
                ev['flight_id'],
                ev['operation'],
                ev['planned_time'].strftime('%Y-%m-%d %H:%M:%S'),
                ev['scheduled_time'].strftime('%Y-%m-%d %H:%M:%S'),
                ev.get('runway', ''),
                f"{delay:.4f}"
            ])

    # ---- 2. 统计指标 ----
    delays = [(ev['scheduled_time'] - ev['planned_time']).total_seconds() / 60
              for ev in schedule]
    delayed = [d for d in delays if d > 0]
    early   = [d for d in delays if d < 0]
    on_time = [d for d in delays if d == 0]

    avg_delay     = sum(delayed) / len(delayed) if delayed else 0.0
    max_delay     = max(delayed) if delayed else 0.0
    avg_early     = sum(early) / len(early) if early else 0.0
    max_early     = min(early) if early else 0.0   # 最大提前（负值最小）
    delay_ratio   = len(delayed) / len(delays) * 100
    early_ratio   = len(early)   / len(delays) * 100
    on_time_ratio = len(on_time) / len(delays) * 100

    # ---- 3. 验证跑道安全间隔约束 ----
    safety_interval = metadata['safety_interval']
    runway_violations = 0
    # 按跑道分组，检查同跑道相邻航班间隔
    from collections import defaultdict
    runway_events = defaultdict(list)
    for ev in schedule:
        runway_events[ev.get('runway', 0)].append(ev['scheduled_time'])
    for rw, times in runway_events.items():
        times_sorted = sorted(times)
        for i in range(len(times_sorted) - 1):
            gap = (times_sorted[i+1] - times_sorted[i]).total_seconds() / 60
            if gap < safety_interval:
                runway_violations += 1

    # ---- 4. 保存 JSON 摘要 ----
    summary = {
        'algorithm': result['algorithm'],
        'timestamp': timestamp,
        'airport': metadata['airport'],
        'n_runways': metadata['n_runways'],
        'safety_interval_minutes': safety_interval,
        'total_flights': len(schedule),
        'generations': result['generations'],
        'total_penalty': round(result['penalty'], 4),
        'delay_stats': {
            'delayed_flights': len(delayed),
            'early_flights': len(early),
            'on_time_flights': len(on_time),
            'delay_ratio_pct': round(delay_ratio, 2),
            'early_ratio_pct': round(early_ratio, 2),
            'on_time_ratio_pct': round(on_time_ratio, 2),
            'avg_delay_minutes': round(avg_delay, 4),
            'max_delay_minutes': round(max_delay, 4),
            'avg_early_minutes': round(avg_early, 4),
            'max_early_minutes': round(max_early, 4),
        },
        'constraint_violations': {
            'runway_interval_violations': runway_violations
        }
    }
    json_path = os.path.join(results_dir, f"schedule_summary_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- 5. 控制台输出 ----
    print("\n" + "=" * 70)
    print("Results Validation & Statistics")
    print("=" * 70)
    print(f"  Delayed flights:  {len(delayed):>6}  ({delay_ratio:.1f}%)")
    print(f"  Early flights:    {len(early):>6}  ({early_ratio:.1f}%)")
    print(f"  On-time flights:  {len(on_time):>6}  ({on_time_ratio:.1f}%)")
    print(f"  Avg delay:        {avg_delay:.2f} minutes")
    print(f"  Max delay:        {max_delay:.2f} minutes")
    print(f"  Avg early:        {abs(avg_early):.2f} minutes")
    print(f"  Max early:        {abs(max_early):.2f} minutes")
    print(f"\n  Runway interval violations: {runway_violations}")
    if runway_violations == 0:
        print("  ✓ All runway safety interval constraints satisfied")
    else:
        print(f"  ✗ {runway_violations} interval violation(s) found (< {safety_interval} minutes)")
    print(f"\n  Results saved:")
    print(f"    {csv_path}")
    print(f"    {json_path}")

if __name__ == "__main__":
    test_ga()
