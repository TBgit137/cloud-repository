"""
蚁群算法模块
用于优化航班跑道调度
"""

import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from .objective_function import ObjectiveFunction
from .constraints import RunwayConstraints

# 尝试导入Numba加速
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("提示: 安装numba可显著加速算法运行 (pip install numba)")


class AntColonyAlgorithm:
    """
    蚁群算法类：通过信息素引导寻找最优航班排序
    
    编码方式：
    - 节点：每个航班
    - 路径：航班执行顺序
    - 信息素：好的排序会留下更多信息素
    """
    
    def __init__(self,
                 n_ants: int = 30,
                 n_iterations: int = 100,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 evaporation_rate: float = 0.5,
                 q: float = 100,
                 n_runways: int = 5):
        """
        初始化蚁群算法参数
        
        Args:
            n_ants: 蚂蚁数量
            n_iterations: 迭代次数
            alpha: 信息素重要程度因子
            beta: 启发式因子重要程度
            evaporation_rate: 信息素挥发率
            q: 信息素强度
            n_runways: 机场跑道数量（默认5条）
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.n_runways = n_runways
        
        self.objective_func = ObjectiveFunction()
        self.constraints = RunwayConstraints()
    
    def _initialize_pheromone(self, n_flights: int) -> np.ndarray:
        """
        初始化信息素矩阵
        
        Args:
            n_flights: 航班数量
        
        Returns:
            信息素矩阵
        """
        return np.ones((n_flights, n_flights))
    
    def _calculate_heuristic(self, flights: List[Dict]) -> np.ndarray:
        """
        计算启发式信息（最早计划时间优先）
        
        Args:
            flights: 航班列表
        
        Returns:
            启发式矩阵
        """
        n = len(flights)
        heuristic = np.zeros((n, n))
        
        # 按计划时间排序的优先级
        sorted_indices = sorted(range(n), 
                              key=lambda i: flights[i]['planned_time'])
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 如果j在排序中更靠前，给予更高的启发值
                    priority_diff = sorted_indices.index(i) - sorted_indices.index(j)
                    heuristic[i][j] = 1.0 / (1.0 + max(0, priority_diff))
        
        return heuristic
    
    def _select_next_flight(self, current: int, unvisited: List[int],
                           pheromone: np.ndarray, 
                           heuristic: np.ndarray) -> int:
        """
        选择下一个访问的航班
        
        Args:
            current: 当前航班索引
            unvisited: 未访问航班列表
            pheromone: 信息素矩阵
            heuristic: 启发式矩阵
        
        Returns:
            下一个航班索引
        """
        if not unvisited:
            return -1
        
        # 计算转移概率
        probabilities = []
        for next_flight in unvisited:
            tau = pheromone[current][next_flight] ** self.alpha
            eta = heuristic[current][next_flight] ** self.beta
            probabilities.append(tau * eta)
        
        # 归一化
        total = sum(probabilities)
        if total == 0:
            return np.random.choice(unvisited)
        
        probabilities = [p / total for p in probabilities]
        
        # 轮盘赌选择
        selected_idx = np.random.choice(len(unvisited), p=probabilities)
        return unvisited[selected_idx]
    
    def _construct_solution(self, flights: List[Dict],
                           pheromone: np.ndarray,
                           heuristic: np.ndarray) -> List[int]:
        """
        构造一个解（航班排序）
        
        Args:
            flights: 航班列表
            pheromone: 信息素矩阵
            heuristic: 启发式矩阵
        
        Returns:
            航班索引排序
        """
        n = len(flights)
        unvisited = list(range(n))
        path = []
        
        # 从计划时间最早的航班开始
        earliest_idx = min(range(n), key=lambda i: flights[i]['planned_time'])
        current = earliest_idx
        path.append(current)
        unvisited.remove(current)
        
        # 构造完整路径
        while unvisited:
            next_flight = self._select_next_flight(current, unvisited,
                                                   pheromone, heuristic)
            path.append(next_flight)
            unvisited.remove(next_flight)
            current = next_flight
        
        return path
    
    def _decode_solution(self, path: List[int], 
                        flights: List[Dict]) -> List[Dict]:
        """
        将路径解码为时刻表（支持多跑道并行）
        
        Args:
            path: 航班索引排序
            flights: 航班列表
        
        Returns:
            时刻表
        """
        schedule = []
        # 维护每个跑道的最后使用时间
        runway_last_times = [None] * self.n_runways
        
        for idx in path:
            flight = flights[idx]
            planned_time = flight['planned_time']
            
            # 找到最早可用的跑道
            best_runway = None
            earliest_available_time = None
            
            for runway_id in range(self.n_runways):
                if runway_last_times[runway_id] is None:
                    # 跑道未使用，可以使用计划时间
                    available_time = planned_time
                else:
                    # 跑道已使用，需要等待安全间隔
                    available_time = max(
                        planned_time,
                        runway_last_times[runway_id] + timedelta(
                            minutes=self.constraints.min_interval
                        )
                    )
                
                # 选择最早可用的跑道
                if earliest_available_time is None or available_time < earliest_available_time:
                    earliest_available_time = available_time
                    best_runway = runway_id
            
            # 安排到最佳跑道
            scheduled_time = earliest_available_time
            runway_last_times[best_runway] = scheduled_time
            
            schedule.append({
                'flight_id': flight.get('flight_id', idx),
                'planned_time': planned_time,
                'scheduled_time': scheduled_time,
                'operation': flight.get('operation', 'departure'),
                'runway': best_runway + 1  # 跑道编号从1开始
            })
        
        return schedule
    
    def _update_pheromone(self, pheromone: np.ndarray,
                         all_paths: List[List[int]],
                         all_penalties: List[float]) -> np.ndarray:
        """
        更新信息素
        
        Args:
            pheromone: 当前信息素矩阵
            all_paths: 所有蚂蚁的路径
            all_penalties: 所有路径的惩罚值
        
        Returns:
            更新后的信息素矩阵
        """
        # 挥发
        pheromone *= (1 - self.evaporation_rate)
        
        # 增加信息素
        for path, penalty in zip(all_paths, all_penalties):
            # 惩罚越小，增加的信息素越多
            delta = self.q / (penalty + 1)
            
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i + 1]] += delta
        
        return pheromone
    
    @staticmethod
    def load_preprocessed_data(json_path: str) -> Tuple[List[Dict], Dict]:
        """
        从预处理输出的JSON文件加载数据
        
        Args:
            json_path: 预处理输出的JSON文件路径
        
        Returns:
            (航班列表, 元数据字典)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为算法所需格式
        flights = []
        for event in data['events']:
            flights.append({
                'flight_id': event['flight_id'],
                'planned_time': datetime.fromisoformat(event['scheduled_time']),
                'operation': event['event_type']
            })
        
        metadata = {
            'airport': data['airport'],
            'n_runways': data['n_runways'],
            'safety_interval': data['safety_interval_minutes'],
            'total_events': data['total_events'],
            'departure_events': data['departure_events'],
            'arrival_events': data['arrival_events']
        }
        
        return flights, metadata
    
    def optimize(self, flights: List[Dict]) -> Dict:
        """
        执行蚁群算法优化
        
        Args:
            flights: 航班列表
        
        Returns:
            优化结果字典
        """
        n_flights = len(flights)
        
        # 初始化
        pheromone = self._initialize_pheromone(n_flights)
        heuristic = self._calculate_heuristic(flights)
        
        best_schedule = None
        best_penalty = float('inf')
        penalty_history = []
        
        for iteration in range(self.n_iterations):
            all_paths = []
            all_penalties = []
            
            # 每只蚂蚁构造解
            for ant in range(self.n_ants):
                path = self._construct_solution(flights, pheromone, heuristic)
                schedule = self._decode_solution(path, flights)
                penalty = self.objective_func.calculate_total_penalty(schedule)
                
                all_paths.append(path)
                all_penalties.append(penalty)
                
                # 更新最优解
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_schedule = schedule
            
            # 更新信息素
            pheromone = self._update_pheromone(pheromone, all_paths, 
                                              all_penalties)
            
            penalty_history.append(best_penalty)
        
        return {
            'algorithm': 'Ant Colony Algorithm',
            'schedule': best_schedule,
            'penalty': best_penalty,
            'penalty_history': penalty_history,
            'iterations': self.n_iterations
        }
