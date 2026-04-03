# 航班跑道智能调度系统 技术报告

---

## 一、项目概述

本项目是一套面向机场运营场景的航班跑道智能调度系统，采用前后端分离架构。用户通过 Web 界面上传航班历史数据集，配置调度参数，系统自动完成数据清洗、预处理，并调用遗传算法对跑道排班进行优化，最终将排班结果实时推送至前端展示。

系统以巴西航班数据集（Brazilian-flights.csv）为基础数据源，目标机场默认为 SBGR（圣保罗瓜鲁柳斯国际机场），支持用户自定义机场代码、跑道数量、安全间隔及调度时间范围。

---

## 二、目录结构

### 2.1 实际业务结构

```
airline-scheduler/
├── backend/
│   ├── src/
│   │   ├── api/
│   │   │   └── app.py                      # FastAPI 后端服务，唯一对外 API 入口
│   │   ├── data/
│   │   │   ├── preprocessor.py             # 数据预处理器（清洗、时区转换、格式化）
│   │   │   ├── validator.py                # 数据集格式验证
│   │   │   └── __init__.py
│   │   ├── algorithm/
│   │   │   ├── genetic_algorithm.py        # 遗传算法（主要优化算法）
│   │   │   ├── ant_colony_algorithm.py     # 蚁群算法（备用，已从业务流中移除）
│   │   │   ├── particle_swarm_algorithm.py # 粒子群算法（备用，已从业务流中移除）
│   │   │   ├── objective_function.py       # 目标函数（延误惩罚计算）
│   │   │   ├── constraints.py              # 约束条件（跑道安全间隔）
│   │   │   └── __init__.py
│   │   └── output/                         # 运行时输出目录（自动创建）
│   │       ├── preprocessed/               # 预处理结果（CSV + JSON + 处理日志）
│   │       └── results/                    # 排班结果（CSV + 摘要 JSON）
│   └── dataset/
│       └── dataset/
│           ├── processed/
│           │   └── Brazilian-flights.csv   # 主数据集
│           └── new/
│               └── flights.csv             # 备用数据集
└── frontend/
    ├── src/
    │   ├── api/
    │   │   └── index.js                    # 封装后端 API 调用，处理 SSE 流
    │   ├── views/
    │   │   └── SchedulerView.vue           # 主页面视图（唯一路由页面）
    │   ├── components/
    │   │   ├── FileDropZone.vue            # 文件拖拽上传组件
    │   │   └── DateRangePicker.vue         # 日期范围选择组件
    │   ├── App.vue                         # 根组件（仅包含 router-view）
    │   └── main.js                         # 应用入口，路由配置
    ├── public/
    │   └── plane.png                       # 品牌图标
    ├── index.html                          # HTML 入口
    ├── vite.config.js                      # Vite 构建配置
    └── package.json                        # 前端依赖声明
```

### 2.2 实验与调试结构（tests 目录）

tests 目录不参与实际业务流程，是开发阶段用于独立验证各模块功能的脚本集合。

```
backend/tests/
├── test_preprocessor.py
│       独立测试数据预处理流程。可配置 CSV_PATH、AIRPORT_CODE、日期范围等参数，
│       输出清洗后数据和算法输入 JSON 到 output/ 目录。
│
├── test_ga.py
│       独立测试遗传算法。自动读取 output/preprocessed/ 中最新的预处理文件，
│       可调整 population_size、generations 等超参数，输出排班结果 CSV 和摘要
│       JSON，并打印约束验证报告（跑道间隔是否满足、延误统计）。
│
├── test_aco.py
│       独立测试蚁群算法，结构同 test_ga.py。
│
├── test_pso.py
│       独立测试粒子群算法，结构同 test_ga.py。
│
├── run_algorithm_from_log.py
│       多算法对比脚本。自动加载最新预处理数据，依次运行三种算法，
│       输出各算法惩罚值对比排名，便于参数调优和算法选型。
│
├── test_algorithm_with_preprocessed.py
│       指定特定预处理文件运行算法，用于复现特定实验结果。
│
└── evaluate_schedule.py
        排班质量评估脚本。对比原始计划延误 vs 算法调度延误，
        计算改善百分比，输出详细统计报告。
```

tests 目录的核心价值在于将预处理和算法解耦，允许开发者在不启动 Web 服务的情况下单独调试任意模块，并通过 run_algorithm_from_log.py 和 evaluate_schedule.py 对算法效果进行量化评估。

---

## 三、完整业务流程

### 3.1 流程总览

```
用户操作
  │
  ├─ 上传 CSV 文件 + 填写调度参数（机场代码、跑道数、安全间隔、日期范围）
  │
  ▼
前端预检（列头格式验证，本地完成，无需网络请求）
  │
  ├─ 验证失败 → 显示错误，终止
  │
  ▼
POST /api/upload（multipart/form-data）
  │
  ▼
后端 SSE 流开始推送进度
  │
  ├─ 阶段 1：读取文件
  ├─ 阶段 2：验证数据集格式
  ├─ 阶段 3：数据预处理（清洗、时区转换等）
  ├─ 阶段 4：遗传算法优化排班
  ├─ 阶段 5：整理并保存结果
  │
  ▼
SSE done 事件（携带完整结果数据）
  │
  ▼
前端渲染
  ├─ 调度摘要（机场、跑道数、事件数、惩罚值、清洗统计）
  ├─ 排班结果表（支持排序、过滤、分页）
  └─ 已清洗航班表（时段内因异常被清洗的航班）
```

### 3.2 前端预检

用户点击"开始调度"后，前端首先在本地读取 CSV 文件的前 4KB，解析第一行列头，与系统要求的 12 个必要列进行比对：

```
Flight.No, Airport.From, Airport.To, Scheduled.Departure, Scheduled.Arrival,
Departure, Arrival, Distance.In.Meters,
Longitude.From, Latitude.From, Longitude.To, Latitude.To
```

此步骤在本地完成，可快速拦截格式不符的文件，避免无效的网络请求。

### 3.3 后端数据验证（validator.py）

后端接收到文件后进行深度验证，包括：

- 列头完整性检查（12 个必要列）
- 数据集非空检查
- 目标机场在数据集中是否存在
- 时间列可解析率（超过 50% 无法解析则拒绝）
- 经纬度范围合法性（经度 [-180, 180]，纬度 [-90, 90]）

### 3.4 数据预处理（preprocessor.py）

预处理分为 7 个串行步骤，每步均记录结构化日志：

**步骤 1：机场与时间段筛选**
按 Airport.From 或 Airport.To 筛选目标机场，再按 Scheduled.Departure 筛选用户指定的日期范围。筛选后的数据集（df_filtered）作为"时段内原始数据"保留，用于后续计算被清洗航班。

**步骤 2：删除缺失值与重复值**
删除 Scheduled.Departure 或 Scheduled.Arrival 为空的行，以及完全重复的行。

**步骤 3：删除时间逻辑错误**
删除计划起飞时间晚于计划降落时间的航班，以及实际起飞时间晚于实际降落时间的航班。

**步骤 4：本地时间转换为 UTC**
利用 timezonefinder 库，根据起降机场的经纬度自动识别时区，将所有时间字段统一转换为 UTC，生成 Scheduled.Departure.UTC 和 Scheduled.Arrival.UTC 列。UTC 转换失败的行被删除。

**步骤 5：航行速度验证**
计算每个航班的平均速度（距离 / 飞行时间），删除速度低于 50 m/s（约 180 km/h）或高于 300 m/s（约 1080 km/h）的异常航班。

**步骤 6：添加唯一标识**
为清洗后的每条记录生成顺序整数 Flight_ID（从 1 开始）。

**步骤 7：生成算法输入**
将清洗后的数据转换为标准化 JSON 格式，分别提取起飞事件（Airport.From = 目标机场）和降落事件（Airport.To = 目标机场），合并排序后生成 events 列表，每个事件包含 flight_id、event_type、scheduled_time。

预处理完成后，系统将 df_filtered（时段内原始数据）与 df_final（清洗后数据）的差集作为"已清洗航班"返回前端，确保展示的是用户指定时段内因异常被清洗的航班，而非时段外的数据。

### 3.5 遗传算法优化

算法在独立线程池（ThreadPoolExecutor）中运行，不阻塞 FastAPI 事件循环。详见第六节。

### 3.6 结果序列化与持久化

算法完成后，系统将排班结果序列化为 JSON 可传输格式（datetime 转 ISO 字符串，计算 delay_minutes），同时将结果保存至 output/results/ 目录（CSV + 摘要 JSON），供后续离线分析使用。

---

## 四、前端技术特点

### 4.1 技术栈

| 技术 | 用途 |
|------|------|
| Vue 3 Composition API | 前端框架，响应式状态管理 |
| Vite | 构建工具与开发服务器 |
| Vue Router 4 | 客户端路由（createWebHistory） |

### 4.2 SSE 流式进度接收

前端不使用传统的 EventSource API，而是通过 fetch + ReadableStream 手动解析 SSE 流。这样做的关键优势是可以在同一个 POST 请求中携带文件和参数，而 EventSource 仅支持 GET 请求，无法传输文件。

实现要点：

- 使用 `res.body.getReader()` 获取流读取器
- 使用 `TextDecoder` 将字节流解码为字符串，`stream: true` 选项处理多字节字符跨 chunk 的情况
- 维护 buffer 字符串，处理跨 chunk 的不完整消息帧
- 以 `\n\n` 为分隔符解析 SSE 消息帧，解析 `event:` 和 `data:` 字段
- 根据事件类型分发：`status` 追加进度步骤，`error` 触发 reject，`done` 触发 resolve

```javascript
const parts = buffer.split('\n\n')
buffer = parts.pop()  // 保留不完整的尾部，等待下一个 chunk
for (const part of parts) {
  let eventName = 'message', dataStr = ''
  for (const line of part.split('\n')) {
    if (line.startsWith('event: ')) eventName = line.slice(7).trim()
    else if (line.startsWith('data: ')) dataStr = line.slice(6).trim()
  }
  // 根据 eventName 分发处理
}
```

### 4.3 进度步骤可视化

进度列表采用追加式设计，每收到一条 status 事件就向列表末尾追加一条记录。通过 CSS 类动态切换三种视觉状态：

- `progress-step--active`：当前正在执行，显示纯 CSS 旋转动画（spinner）
- `progress-step--done`：已完成，绿色背景 + ✓ 图标
- `progress-step--error`：出错，红色背景 + ✕ 图标

旋转动画通过纯 CSS `@keyframes` 实现，无需引入任何动画库。

### 4.4 内联子组件

为保持单文件组件的简洁性，`SortIcon`（排序图标）和 `Pagination`（分页控件）两个通用 UI 组件直接在 SchedulerView.vue 的 `<script setup>` 中通过 `defineComponent` + `h` 渲染函数定义，避免引入额外文件。

### 4.5 响应式数据表格

排班结果表格支持以下功能，全部在前端内存中完成，无需额外请求：

- **多列排序**：点击列头切换升降序，通过 `reactive` 对象维护 `{ col, asc }` 排序状态
- **关键词搜索**：实时过滤航班号和跑道号
- **类型过滤**：按起飞 / 降落筛选
- **分页**：每页 50 条，通过 `computed` 属性对过滤后的数组切片

### 4.6 文件拖拽上传（FileDropZone.vue）

支持拖拽和点击两种上传方式，通过监听 `dragover`、`dragleave`、`drop` 事件管理拖拽状态，使用 `v-model:file` 双向绑定文件对象，文件选中后显示文件名和大小（自动换算 KB / MB）。

---

## 五、后端技术特点

### 5.1 技术栈

| 技术 | 用途 |
|------|------|
| FastAPI | Web 框架，原生支持异步和 StreamingResponse |
| Uvicorn | ASGI 服务器 |
| Pandas | 数据处理与 CSV 读写 |
| NumPy | 数值计算（遗传算法向量运算） |
| timezonefinder | 根据经纬度自动识别时区 |
| pytz | 时区转换 |
| ThreadPoolExecutor | CPU 密集型任务异步化 |

### 5.2 SSE 流式响应

后端使用 FastAPI 的 `StreamingResponse` 配合异步生成器实现 SSE：

```python
async def event_stream():
    yield sse("status", {"step": "read", "message": "正在读取文件..."})
    # ... 各阶段处理 ...
    yield sse("done", {"summary": ..., "schedule": ..., "removed_flights": ...})

return StreamingResponse(
    event_stream(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```

`X-Accel-Buffering: no` 响应头用于禁用 Nginx 等反向代理的响应缓冲，确保每条 SSE 事件实时到达客户端，而不是被批量缓冲后一次性发送。

### 5.3 CPU 密集型任务异步化

预处理和遗传算法均为 CPU 密集型操作，若直接在 async 函数中同步执行会阻塞整个事件循环，导致 SSE 连接无法推送中间状态。系统使用 `ThreadPoolExecutor` 配合 `loop.run_in_executor` 将其放入线程池执行：

```python
_executor = ThreadPoolExecutor(max_workers=2)

clean_df, algorithm_input, filtered_df = await loop.run_in_executor(
    _executor,
    partial(preprocessor.process, original_df.copy(), airport, ...)
)
```

这样在算法运行期间，FastAPI 事件循环仍可处理其他请求，SSE 连接保持活跃。

### 5.4 CORS 配置

允许来自 `127.0.0.1:8080`、`localhost:8080`、`localhost:5173`、`127.0.0.1:5173` 的跨域请求，覆盖 Vite 开发服务器的常见端口组合。

---

## 六、遗传算法详解

### 6.1 问题建模

跑道调度问题被建模为一个连续优化问题：给定 N 个航班事件（起飞或降落），每个事件有一个计划时间，需要为每个事件分配一个实际执行时间和跑道编号，目标是在满足跑道安全间隔约束的前提下，最小化所有事件的总延误惩罚。

### 6.2 编码方式

采用实数编码：

- **个体（染色体）**：长度为 N 的浮点数数组
- **基因**：第 i 个基因表示第 i 个航班的时间偏移量（分钟），范围 [-max_offset/2, max_offset]
- 正值表示延误，负值表示提前

### 6.3 种群初始化策略

初始种群采用混合策略，兼顾收敛速度和种群多样性：

- **80% 个体**：从准点附近小幅扰动出发（±5 分钟），使算法从接近可行解的状态开始，加快早期收敛
- **20% 个体**：完全随机初始化，防止种群早熟收敛于局部最优

### 6.4 解码与约束处理

个体解码时，先将偏移量转换为初步调度时间，再通过贪心策略满足跑道安全间隔约束：

1. 按调度时间对所有事件排序
2. 维护每条跑道的最后使用时间数组（长度 = n_runways）
3. 对每个事件，遍历所有跑道，找到最早可用时间最小的跑道
4. 将事件分配到该跑道，更新该跑道的最后使用时间
5. 硬约束：调度时间不早于计划时间（不允许提前于计划时间执行）

### 6.5 目标函数（分段非线性惩罚）

#### 6.5.1 设计目标

目标函数的核心任务是将一份排班方案的"质量"量化为一个标量惩罚值，遗传算法以最小化该值为优化目标。设计时需要满足以下要求：

1. **准点无惩罚**：排班时间等于计划时间时，惩罚为 0
2. **非线性递增**：延误越长，惩罚增长越快，体现航空运营中长延误的不可接受性
3. **不鼓励提前**：提前执行在实际运营中同样有害（旅客未到、地面保障未就绪），因此提前的惩罚高于同等时长的延误
4. **分段连续**：各段在边界处连续，保证适应度曲面相对平滑，有利于遗传算法的搜索

#### 6.5.2 惩罚函数类型

系统实现了两种惩罚函数，通过 `penalty_type` 参数切换，默认使用分段函数。

**（1）分段幂函数（piecewise，默认）**

设偏差 δ = |scheduled_time − planned_time|（单位：分钟），基础惩罚值按以下四段计算：

```
δ = 0：
    base = 0

0 < δ ≤ 10（轻微偏差，线性增长）：
    base = δ

10 < δ ≤ 30（中度延误，加速增长）：
    base = 10 + 3 × (δ − 10)^1.5

30 < δ ≤ 60（重度延误，陡化增长）：
    base = 10 + 3×(20^1.5) + 8 × (δ − 30)^1.8

δ > 60（严重延误，二次增长）：
    base = 10 + 3×(20^1.5) + 8×(30^1.8) + 15 × (δ − 60)^2.0
```

各段的系数（3、8、15）和指数（1.5、1.8、2.0）均逐段递增，确保惩罚曲线在分段边界处连续且单调递增，越靠后的延误段惩罚增长越陡峭。

最终惩罚值引入方向系数，区分提前与延误：

```
penalty = multiplier × base

其中：
  multiplier = 3.0，若 delay_minutes < 0（提前执行）
  multiplier = 1.0，若 delay_minutes ≥ 0（延误或准点）
```

提前系数 3.0 意味着提前 10 分钟的惩罚等同于延误 10 分钟惩罚的 3 倍，强烈驱动算法不将航班安排在计划时间之前。这一设计反映了实际运营约束：提前起飞可能导致旅客误机，提前降落可能导致停机位冲突。

**（2）指数函数（exponential，备用）**

```
penalty = 0.05 × δ^2.2
```

形式更简洁，对所有偏差统一使用 2.2 次幂增长，但缺乏分段控制，对小偏差的容忍度较低。适合对准点率要求极高、不需要区分轻微/严重延误的场景。

#### 6.5.3 典型惩罚值对照

以下为分段函数（延误方向，multiplier = 1）的典型取值，便于直观理解各段的惩罚量级：

| 偏差（分钟） | 惩罚值（约） | 所在段 |
|------------|------------|--------|
| 0 | 0 | 准点 |
| 5 | 5.0 | 线性段 |
| 10 | 10.0 | 线性段上限 |
| 20 | ≈ 104.9 | 加速段 |
| 30 | ≈ 283.1 | 加速段上限 |
| 45 | ≈ 1,148 | 重度段 |
| 60 | ≈ 2,283 | 重度段上限 |
| 90 | ≈ 15,783 | 极重段 |

提前方向的惩罚值为上表对应值的 3 倍（例如提前 10 分钟 ≈ 30.0）。

从数值可以看出，60 分钟以上的延误惩罚约为 10 分钟延误的 2,283 倍，这种极端非线性设计使遗传算法在进化过程中会优先消除大延误，而非均匀分散延误。

#### 6.5.4 总惩罚计算

对一份包含 N 个航班事件的排班方案，总惩罚为所有事件惩罚值的简单求和：

```
TotalPenalty = Σ penalty(scheduled_time_i − planned_time_i)，i = 1..N
```

遗传算法的适应度值定义为总惩罚的负值（`fitness = −TotalPenalty`），使得惩罚越小、适应度越高，符合遗传算法"适应度越大越好"的选择逻辑。

#### 6.5.5 辅助方法

`ObjectiveFunction` 类还提供以下辅助方法，主要用于实验分析和结果评估：

- `evaluate_schedule(departures, arrivals)`：分别计算起飞和降落的惩罚，返回三项统计（起飞惩罚、降落惩罚、总惩罚），便于分析起降两类事件的优化效果差异
- `get_penalty_curve(max_delay_minutes)`：生成惩罚曲线数据（二维数组 [[延误时间, 惩罚值], ...]），可用于可视化惩罚函数形状，辅助参数调优
- `get_penalty_statistics(delays_minutes)`：对一组延误时间列表计算统计信息，包括平均延误、最大延误、最小延误、总惩罚、平均惩罚、最大惩罚
### 6.6 遗传操作

**选择**：轮盘赌选择，将适应度值归一化为概率分布后随机采样。

**交叉**：单点交叉，以 crossover_rate（默认 0.8）的概率执行，随机选择交叉点生成两个子代。

**变异**：对每个基因以 mutation_rate（默认 0.3）的概率执行变异：
- 80% 概率：微调（±15 分钟），在当前值附近局部搜索
- 20% 概率：随机重置到 [-max_offset/2, max_offset]，跳出局部最优
- 额外操作：以 mutation_rate 的概率随机交换相邻两个基因，增加排列多样性

**精英保留**：每代保留适应度最高的 elite_size（默认 2）个个体，防止最优解因随机操作退化。

### 6.7 算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| population_size | 50 | 种群大小 |
| generations | 100 | 迭代代数 |
| crossover_rate | 0.8 | 交叉概率 |
| mutation_rate | 0.3 | 变异概率 |
| elite_size | 2 | 精英保留数量 |
| max_offset | 60 | 最大时间偏移（分钟） |
| n_runways | 5 | 跑道数量（可由用户配置覆盖） |

---

## 七、组件交互细节

### 7.1 前后端交互时序

```
前端                                    后端
 │                                       │
 │── POST /api/upload ──────────────────>│
 │   (multipart: file + params)          │
 │                                       │── yield sse("status", "读取文件...")
 │<── SSE: status(read) ─────────────────│
 │                                       │── validate(df, airport)
 │<── SSE: status(validate) ─────────────│
 │                                       │── preprocessor.process() [线程池]
 │<── SSE: status(preprocess) ───────────│
 │<── SSE: status(preprocess_done) ──────│
 │                                       │── _run_ga() [线程池]
 │<── SSE: status(optimize) ─────────────│
 │                                       │── _serialize_schedule()
 │                                       │── _get_removed_flights()
 │<── SSE: status(finalize) ─────────────│
 │                                       │── 保存 CSV + JSON 到 output/
 │<── SSE: done(summary + schedule       │
 │           + removed_flights) ─────────│
 │                                       │
 │── 渲染排班表 + 清洗航班表               │
```

### 7.2 前端组件数据流

```
SchedulerView.vue
  │
  ├── FileDropZone.vue
  │     emit('update:file') ──────────> form.file
  │
  ├── DateRangePicker.vue
  │     emit('update:modelValue') ────> form.dateRange
  │
  ├── onSubmit()
  │     │
  │     ├── checkHeaders(form.file)          [本地列头验证]
  │     │
  │     └── uploadDataset(file, params, onStatus)  [api/index.js]
  │           │
  │           ├── onStatus(step, message) ──> pushStep() ──> steps[]
  │           │                                              （进度列表）
  │           └── resolve(result)
  │                 ├── result.summary ──────> summary（摘要数据）
  │                 ├── result.schedule ─────> schedule[]（排班数据）
  │                 └── result.removed_flights > removedFlights[]
  │
  ├── filteredSchedule (computed)
  │     输入：schedule[] + scheduleSearch + scheduleOpFilter + sort
  │     输出：过滤并排序后的数组
  │
  └── pagedSchedule (computed)
        输入：filteredSchedule + schedulePage + PAGE_SIZE(50)
        输出：当前页的数据切片
```

### 7.3 后端模块依赖关系

```
app.py（API 入口）
  │
  ├── validator.py
  │     validate(df, airport_code) → (bool, error_message)
  │
  ├── preprocessor.py
  │     FlightDataPreprocessor.process(df, ...) → (clean_df, algorithm_input, filtered_df)
  │       内部依赖：
  │         timezonefinder  → 经纬度识别时区
  │         pytz            → 时区转换
  │
  └── genetic_algorithm.py
        GeneticAlgorithm.optimize(flights) → result_dict
          内部依赖：
            objective_function.py
              ObjectiveFunction.calculate_total_penalty(schedule) → float
            constraints.py
              RunwayConstraints.min_interval → float
```

### 7.4 输出文件结构

每次运行后，系统在 `src/output/` 目录下生成带时间戳的文件：

```
src/output/
├── preprocessed/
│   ├── cleaned_data_YYYYMMDD_HHMMSS.csv
│   │     清洗后的完整航班数据，包含所有原始列 + UTC 时间列 + Flight_ID
│   ├── algorithm_input_YYYYMMDD_HHMMSS.json
│   │     算法输入，包含 airport、n_runways、safety_interval_minutes、events[]
│   └── processing_log_YYYYMMDD_HHMMSS.json
│         各步骤处理日志，记录每步的输入行数、输出行数、删除原因
└── results/
    ├── schedule_result_YYYYMMDD_HHMMSS.csv
    │     排班结果，包含 flight_id、operation、planned_time、scheduled_time、
    │     delay_minutes、runway
    └── schedule_summary_YYYYMMDD_HHMMSS.json
          摘要，包含 algorithm、penalty、total_scheduled、airport、
          n_runways、safety_interval
```

---

## 八、数据集说明

原始数据集为巴西国内航班数据（Brazilian-flights.csv），包含以下关键字段：

| 字段 | 说明 |
|------|------|
| Flight.No | 航班号 |
| Airport.From | 出发机场代码（ICAO） |
| Airport.To | 到达机场代码（ICAO） |
| Scheduled.Departure | 计划起飞时间（本地时间） |
| Scheduled.Arrival | 计划降落时间（本地时间） |
| Departure | 实际起飞时间 |
| Arrival | 实际降落时间 |
| Distance.In.Meters | 航线距离（米） |
| Longitude.From / Latitude.From | 出发机场经纬度 |
| Longitude.To / Latitude.To | 到达机场经纬度 |

---

## 九、运行环境与启动方式

### 9.1 后端

主要依赖：`fastapi`、`uvicorn`、`pandas`、`numpy`、`timezonefinder`、`pytz`

```bash
cd airline-scheduler/backend
uvicorn src.api.app:app --reload --port 8000
```

### 9.2 前端

主要依赖：`vue`、`vue-router`、`vite`、`@vitejs/plugin-vue`

```bash
cd airline-scheduler/frontend
npm install
npm run dev
```

访问地址：`http://127.0.0.1:8080`

---

*本报告基于项目当前代码状态生成，版本日期：2026 年 3 月*
