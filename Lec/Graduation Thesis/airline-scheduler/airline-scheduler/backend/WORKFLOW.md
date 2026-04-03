# 航班跑道调度系统工作流程

## 完整工作流程

### 步骤1: 数据预处理

运行预处理程序，清洗数据并生成算法输入：

```bash
cd airline-scheduler/backend/tests
python test_preprocessor.py
```

或者在代码中调用：

```python
from data import preprocess_flight_data

df_clean, algorithm_input = preprocess_flight_data(
    csv_path="../dataset/dataset/processed/Brazilian-flights.csv",
    airport_code="SBGR",           # 机场代码
    n_runways=5,                   # 跑道数量
    safety_interval=3,             # 安全间隔（分钟）
    start_date="2016-01-01",       # 开始日期
    end_date="2016-01-31",         # 结束日期
    log_dir="../src/log"           # 输出目录
)
```

**输出文件**（保存在 `src/log/` 目录）:
- `cleaned_data_YYYYMMDD_HHMMSS.csv` - 清洗后的完整数据
- `algorithm_input_YYYYMMDD_HHMMSS.json` - 算法输入数据
- `processing_log_YYYYMMDD_HHMMSS.json` - 处理日志

### 步骤2: 运行算法

#### 方法A: 使用便捷脚本（推荐用于实验）

```bash
cd airline-scheduler/backend/tests
python run_algorithm_from_log.py
```

这个脚本会：
1. 自动找到最新的预处理文件
2. 加载数据
3. 运行三种算法
4. 对比结果

#### 方法B: 手动运行单个算法

```python
from algorithm.genetic_algorithm import GeneticAlgorithm

# 加载预处理数据
flights, metadata = GeneticAlgorithm.load_preprocessed_data(
    '../src/log/algorithm_input_20260318_120000.json'
)

# 初始化算法
ga = GeneticAlgorithm(
    population_size=50,
    generations=100,
    n_runways=metadata['n_runways']
)

# 运行优化
result = ga.optimize(flights)

# 查看结果
print(f"总延误: {result['penalty']:.2f} 分钟")
print(f"调度了 {len(result['schedule'])} 个航班")
```

### 步骤3: 分析结果

```python
# 查看调度结果
for event in result['schedule'][:10]:
    print(f"航班 {event['flight_id']}: "
          f"{event['planned_time']} -> {event['scheduled_time']}, "
          f"跑道 {event['runway']}")

# 查看收敛曲线
import matplotlib.pyplot as plt

plt.plot(result['fitness_history'])
plt.xlabel('Generation')
plt.ylabel('Total Penalty (minutes)')
plt.title('Algorithm Convergence')
plt.show()
```

## 目录结构

```
airline-scheduler/backend/
├── dataset/
│   └── dataset/
│       └── processed/
│           └── Brazilian-flights.csv    # 原始数据
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessor.py              # 数据预处理模块
│   │   └── README.md
│   ├── algorithm/
│   │   ├── __init__.py
│   │   ├── genetic_algorithm.py         # 遗传算法
│   │   ├── ant_colony_algorithm.py      # 蚁群算法
│   │   ├── particle_swarm_algorithm.py  # 粒子群算法
│   │   ├── objective_function.py        # 目标函数
│   │   ├── constraints.py               # 约束条件
│   │   └── README.md
│   └── log/                             # 输出目录
│       ├── cleaned_data_*.csv
│       ├── algorithm_input_*.json
│       └── processing_log_*.json
└── tests/
    ├── test_preprocessor.py             # 测试预处理
    ├── test_algorithm_with_preprocessed.py  # 测试算法加载
    └── run_algorithm_from_log.py        # 便捷运行脚本
```

## 数据流

```
原始CSV数据
    ↓
[数据预处理模块]
    ↓
algorithm_input.json (保存在 log/)
    ↓
[算法模块加载]
    ↓
优化结果
```

## 实验调参流程

1. **运行预处理**（只需运行一次）:
```bash
python test_preprocessor.py
```

2. **修改算法参数**:
编辑 `run_algorithm_from_log.py` 中的参数：
```python
ga_params = {
    'population_size': 50,   # 修改这里
    'generations': 100       # 修改这里
}
```

3. **运行算法**:
```bash
python run_algorithm_from_log.py
```

4. **对比结果**，重复步骤2-3

## 常见问题

### Q: 找不到预处理文件
A: 先运行 `test_preprocessor.py` 生成预处理数据

### Q: 算法运行太慢
A: 减少迭代次数或种群大小，或使用更小的数据集（缩短日期范围）

### Q: 如何使用不同的机场
A: 修改 `test_preprocessor.py` 中的 `AIRPORT_CODE` 参数

### Q: 如何调整跑道数量
A: 修改 `test_preprocessor.py` 中的 `N_RUNWAYS` 参数

## 性能建议

### 小规模测试（快速验证）
- 日期范围: 1-3天
- 算法参数: generations/iterations = 20-50

### 中等规模实验
- 日期范围: 1周
- 算法参数: generations/iterations = 50-100

### 大规模实验（完整评估）
- 日期范围: 1个月
- 算法参数: generations/iterations = 100-200

## 下一步开发

- [ ] 添加结果可视化
- [ ] 实现算法结果保存
- [ ] 添加多机场对比
- [ ] 实现主控脚本（整合预处理+算法）
- [ ] 添加实时调度模式
