# 航班跑道调度算法模块

## 概述

本模块实现了三种优化算法用于航班跑道调度：
- 遗传算法 (Genetic Algorithm, GA)
- 蚁群算法 (Ant Colony Optimization, ACO)
- 粒子群算法 (Particle Swarm Optimization, PSO)

所有算法都支持多跑道并行调度，并使用相同的目标函数和约束条件。

## 使用方法

### 方法1: 从预处理JSON文件加载数据（推荐用于实验）

```python
from algorithm.genetic_algorithm import GeneticAlgorithm

# 加载预处理数据
flights, metadata = GeneticAlgorithm.load_preprocessed_data(
    'path/to/algorithm_input.json'
)

# 使用元数据中的跑道数初始化算法
ga = GeneticAlgorithm(
    population_size=50,
    generations=100,
    n_runways=metadata['n_runways']
)

# 运行优化
result = ga.optimize(flights)

print(f"总延误: {result['penalty']:.2f} 分钟")
print(f"调度结果: {result['schedule']}")
```

### 方法2: 直接传入航班数据

```python
from algorithm.genetic_algorithm import GeneticAlgorithm
from datetime import datetime

# 准备航班数据
flights = [
    {
        'flight_id': 1,
        'planned_time': datetime(2026, 3, 18, 8, 0),
        'operation': 'departure'
    },
    {
        'flight_id': 2,
        'planned_time': datetime(2026, 3, 18, 8, 5),
        'operation': 'arrival'
    }
]

# 初始化并运行算法
ga = GeneticAlgorithm(n_runways=5)
result = ga.optimize(flights)
```

## 算法参数

### 遗传算法 (GeneticAlgorithm)

```python
GeneticAlgorithm(
    population_size=50,    # 种群大小
    generations=100,       # 迭代代数
    crossover_rate=0.8,    # 交叉概率
    mutation_rate=0.2,     # 变异概率
    elite_size=5,          # 精英保留数量
    max_offset=60,         # 最大时间偏移量（分钟）
    n_runways=5            # 跑道数量
)
```

### 蚁群算法 (AntColonyAlgorithm)

```python
AntColonyAlgorithm(
    n_ants=30,                  # 蚂蚁数量
    n_iterations=100,           # 迭代次数
    alpha=1.0,                  # 信息素重要程度因子
    beta=2.0,                   # 启发式因子重要程度
    evaporation_rate=0.5,       # 信息素挥发率
    q=100,                      # 信息素强度
    n_runways=5                 # 跑道数量
)
```

### 粒子群算法 (ParticleSwarmAlgorithm)

```python
ParticleSwarmAlgorithm(
    n_particles=30,        # 粒子数量
    n_iterations=100,      # 迭代次数
    w=0.7,                 # 惯性权重
    c1=1.5,                # 个体学习因子
    c2=1.5,                # 社会学习因子
    max_velocity=20.0,     # 最大速度（分钟）
    max_offset=60.0,       # 最大时间偏移量（分钟）
    n_runways=5            # 跑道数量
)
```

## 输入数据格式

### 预处理JSON格式

```json
{
  "airport": "SBGR",
  "n_runways": 5,
  "safety_interval_minutes": 3,
  "total_events": 150,
  "departure_events": 80,
  "arrival_events": 70,
  "events": [
    {
      "flight_id": 1,
      "event_type": "departure",
      "scheduled_time": "2026-03-18T08:00:00",
      "airport": "SBGR"
    }
  ]
}
```

### 航班列表格式

```python
flights = [
    {
        'flight_id': int,           # 航班唯一标识
        'planned_time': datetime,   # 计划时间
        'operation': str            # 'departure' 或 'arrival'
    }
]
```

## 输出结果格式

所有算法返回相同格式的结果字典：

```python
{
    'algorithm': str,              # 算法名称
    'schedule': List[Dict],        # 优化后的时刻表
    'penalty': float,              # 总延误惩罚值（分钟）
    'fitness_history': List[float] # 或 'penalty_history'，每代的最优值
}
```

### 时刻表格式

```python
schedule = [
    {
        'flight_id': int,              # 航班ID
        'planned_time': datetime,      # 计划时间
        'scheduled_time': datetime,    # 调度后时间
        'operation': str,              # 操作类型
        'runway': int                  # 分配的跑道编号（1-n）
    }
]
```

## 约束条件

1. **安全间隔约束**: 同一跑道上相邻航班间隔 ≥ 3分钟（默认）
2. **多跑道并行**: 支持多条跑道同时运作
3. **时间窗口**: 航班可在计划时间前后一定范围内调整

## 目标函数

使用分段惩罚函数，延误时间越长惩罚越重：
- 0-10分钟: 线性惩罚
- 10-30分钟: 1.5次方惩罚
- 30-60分钟: 1.8次方惩罚
- 60分钟以上: 2次方惩罚
- 提前到达: 无惩罚

## 测试脚本

### 基础测试
```bash
cd airline-scheduler/backend/tests
python test_algorithm_with_preprocessed.py
```

### 从最新预处理数据运行
```bash
cd airline-scheduler/backend/tests
python run_algorithm_from_log.py
```

## 调参建议

### 小规模问题（< 100个航班）
- GA: population_size=30, generations=50
- ACO: n_ants=20, n_iterations=50
- PSO: n_particles=30, n_iterations=50

### 中等规模问题（100-500个航班）
- GA: population_size=50, generations=100
- ACO: n_ants=30, n_iterations=100
- PSO: n_particles=50, n_iterations=100

### 大规模问题（> 500个航班）
- GA: population_size=100, generations=200
- ACO: n_ants=50, n_iterations=200
- PSO: n_particles=100, n_iterations=200

## 算法特点

### 遗传算法
- 优点: 全局搜索能力强，适合复杂问题
- 缺点: 收敛速度较慢
- 适用: 大规模、复杂约束问题

### 蚁群算法
- 优点: 适合排序问题，收敛稳定
- 缺点: 参数敏感
- 适用: 中等规模问题

### 粒子群算法
- 优点: 收敛速度快，实现简单
- 缺点: 容易陷入局部最优
- 适用: 快速求解、实时调度

## 文件结构

```
algorithm/
├── __init__.py                    # 模块初始化
├── genetic_algorithm.py           # 遗传算法
├── ant_colony_algorithm.py        # 蚁群算法
├── particle_swarm_algorithm.py    # 粒子群算法
├── objective_function.py          # 目标函数
├── constraints.py                 # 约束条件
└── README.md                      # 本文档
```

## 与预处理模块的集成

1. 运行预处理生成JSON文件:
```python
from data import preprocess_flight_data

df_clean, algorithm_input = preprocess_flight_data(
    csv_path="data.csv",
    airport_code="SBGR",
    n_runways=5,
    safety_interval=3
)
# 输出保存在 log/algorithm_input_*.json
```

2. 从JSON文件加载并运行算法:
```python
from algorithm.genetic_algorithm import GeneticAlgorithm

flights, metadata = GeneticAlgorithm.load_preprocessed_data(
    'log/algorithm_input_20260318_120000.json'
)

ga = GeneticAlgorithm(n_runways=metadata['n_runways'])
result = ga.optimize(flights)
```

## 依赖

- numpy
- datetime (标准库)
- json (标准库)
