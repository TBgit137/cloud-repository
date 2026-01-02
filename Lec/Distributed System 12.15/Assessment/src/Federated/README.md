# 联邦学习模型 - Fashion-MNIST

基于Fashion-MNIST数据集的联邦学习实现，使用FedAvg算法进行模型聚合。

## 项目结构

```
Federal/
├── data_partition.py       # 数据划分模块（IID和Non-IID）
├── client.py               # 客户端类（模拟设备）
├── server.py               # 参数服务器（FedAvg聚合）
├── model.py                # 神经网络模型
├── federated_train.py      # 联邦学习训练脚本
├── federated_evaluate.py   # 评估脚本
├── requirements.txt        # 依赖包
└── README.md               # 本文件
```

## 核心组件

### 1. 数据划分 (data_partition.py)

支持三种数据划分方式：

- **IID (独立同分布)**: 数据随机均匀分配，每个客户端数据分布相同
- **Non-IID (非独立同分布)**: 按标签排序后分片，每个客户端只有部分类别
- **Dirichlet分布**: 使用Dirichlet分布控制数据异质性程度

### 2. 客户端 (client.py)

`FederatedClient` 类模拟联邦学习中的设备：
- 存储本地数据
- 接收全局模型参数
- 本地训练
- 发送更新后的参数

### 3. 参数服务器 (server.py)

`ParameterServer` 类实现：
- 全局模型管理
- FedAvg聚合算法（按样本数量加权平均）
- 模型分发
- 全局模型评估

### 4. 联邦学习流程 (federated_train.py)

完整的联邦学习训练循环：
1. 服务器分发全局模型
2. 客户端本地训练
3. 客户端上传模型参数
4. 服务器聚合参数
5. 重复以上步骤

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python federated_train.py
```

### 3. 配置参数

在 `federated_train.py` 中修改：

```python
NUM_CLIENTS = 10          # 客户端数量
PARTITION_TYPE = 'iid'    # 数据划分: 'iid', 'non_iid', 'dirichlet'
NUM_ROUNDS = 50           # 通信回合数
LOCAL_EPOCHS = 1          # 本地训练轮数
LEARNING_RATE = 0.01      # 学习率
BATCH_SIZE = 32           # 批大小
CLIENT_FRACTION = 1.0     # 每轮参与的客户端比例
```

### 4. 评估模型

训练完成后，修改 `federated_evaluate.py` 中的模型路径，然后运行：

```bash
python federated_evaluate.py
```

## FedAvg算法

联邦平均(Federated Averaging)算法步骤：

1. **初始化**: 服务器初始化全局模型 $w_0$

2. **每轮通信**:
   - 服务器选择客户端子集 $S_t$
   - 服务器将全局模型 $w_t$ 发送给选中的客户端
   - 每个客户端 $k$ 在本地数据上训练，得到 $w_t^k$
   - 服务器聚合: $w_{t+1} = \sum_{k \in S_t} \frac{n_k}{n} w_t^k$
   
   其中 $n_k$ 是客户端 $k$ 的样本数，$n$ 是总样本数

## 数据划分方式对比

| 划分方式 | 特点 | 适用场景 |
|---------|------|---------|
| IID | 数据均匀分布 | 理想情况基准 |
| Non-IID | 每个客户端只有部分类别 | 模拟真实场景 |
| Dirichlet | 可控的异质性程度 | 研究数据异质性影响 |

## 预期性能

- IID划分: 准确率接近集中式模型 (~93%)
- Non-IID划分: 准确率可能略低 (~88-92%)

## 输出文件

训练完成后，在 `logs/` 目录下生成：
- `best_model.pth`: 最佳全局模型
- `history.json`: 训练历史（每轮准确率和损失）

## 与集中式模型对比

集中式模型基准：
- 准确率: 93.02%
- 精确率: 93.05%
- F1分数: 93.03%

联邦学习目标：超越或接近这些指标。
