# 数据预处理模块

## 功能说明

该模块负责清洗和准备航班数据，为调度算法提供标准化输入。

## 主要功能

1. **机场和时间段筛选**: 根据指定机场和日期范围筛选航班
2. **数据清洗**: 删除缺失值、重复值和时间逻辑错误
3. **时区转换**: 根据机场经纬度自动识别时区并转换为UTC
4. **航行时间验证**: 删除速度异常的航班（< 50 m/s 或 > 300 m/s）
5. **唯一标识**: 为每个航班生成UUID
6. **算法输入准备**: 生成标准化的算法输入格式
7. **日志记录**: 保存详细的处理日志和清洗后的数据

## 使用方法

### 方法1: 使用便捷函数

```python
from data import preprocess_flight_data

# 预处理数据
df_clean, algorithm_input = preprocess_flight_data(
    csv_path="path/to/Brazilian-flights.csv",
    airport_code="SBGR",           # 机场代码
    n_runways=5,                   # 跑道数量
    safety_interval=3,             # 安全间隔（分钟）
    start_date="2016-01-01",       # 开始日期（可选）
    end_date="2016-01-31",         # 结束日期（可选）
    log_dir="./log"                # 日志目录
)

# 使用清洗后的数据
print(f"清洗后航班数: {len(df_clean)}")
print(f"总事件数: {algorithm_input['total_events']}")
```

### 方法2: 使用类接口

```python
from data import FlightDataPreprocessor
import pandas as pd

# 加载数据
df = pd.read_csv("path/to/Brazilian-flights.csv", encoding='latin1')

# 创建预处理器
preprocessor = FlightDataPreprocessor(log_dir="./log")

# 执行预处理
df_clean, algorithm_input = preprocessor.process(
    df=df,
    airport_code="SBGR",
    n_runways=5,
    safety_interval=3,
    start_date="2016-01-01",
    end_date="2016-01-31"
)
```

## 输入数据格式

CSV文件需包含以下列：
- `Flight.No`: 航班号
- `Airport.From`: 出发机场
- `Airport.To`: 到达机场
- `Scheduled.Departure`: 计划起飞时间
- `Scheduled.Arrival`: 计划到达时间
- `Departure`: 实际起飞时间（可选）
- `Arrival`: 实际到达时间（可选）
- `Longitude.From`: 出发机场经度
- `Latitude.From`: 出发机场纬度
- `Longitude.To`: 到达机场经度
- `Latitude.To`: 到达机场纬度
- `Distance.In.Meters`: 航程距离（米）

## 输出格式

### 1. 清洗后的数据 (DataFrame)

包含原始列 + 以下新增列：
- `Flight_ID`: 唯一标识（UUID）
- `Departure_TZ`: 出发机场时区
- `Arrival_TZ`: 到达机场时区
- `Scheduled.Departure.UTC`: UTC计划起飞时间
- `Scheduled.Arrival.UTC`: UTC计划到达时间
- `Departure.UTC`: UTC实际起飞时间（如果有）
- `Arrival.UTC`: UTC实际到达时间（如果有）

### 2. 算法输入 (Dict)

```json
{
  "airport": "SBGR",
  "n_runways": 5,
  "safety_interval_minutes": 3,
  "total_events": 22729,
  "departure_events": 11921,
  "arrival_events": 10808,
  "events": [
    {
      "flight_id": "uuid-string",
      "event_type": "departure",  // 或 "arrival"
      "scheduled_time": "2016-01-01T02:00:00",
      "airport": "SBGR"
    },
    ...
  ]
}
```

## 日志文件

处理完成后会在日志目录生成以下文件：
- `cleaned_data_YYYYMMDD_HHMMSS.csv`: 清洗后的完整数据
- `algorithm_input_YYYYMMDD_HHMMSS.json`: 算法输入数据
- `processing_log_YYYYMMDD_HHMMSS.json`: 详细处理日志

## 数据清洗规则

1. **机场筛选**: 起飞机场或降落机场为指定机场
2. **时间筛选**: 计划起飞时间在指定日期范围内
3. **缺失值**: 删除计划起飞/降落时间缺失的航班
4. **时间逻辑**: 删除起飞时间晚于降落时间的航班
5. **速度验证**: 删除平均速度 < 50 m/s 或 > 300 m/s 的航班

## 测试

运行测试脚本：
```bash
cd airline-scheduler/backend/tests
python test_preprocessor.py
```

## 依赖

- pandas
- numpy
- timezonefinder
- pytz
