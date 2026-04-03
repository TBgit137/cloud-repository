"""
遗传算法模块
用于优化航班跑道调度
"""

import random
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from .objective_function import ObjectiveFunction
from .constraints import RunwayConstraints

# 尝试导入Numba加速
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("提示: 安装numba可显著加速算法运行 (pip install numba)")


class GeneticAlgorithm:
    """
    遗传算法类：通过进化策略优化航班时刻表
    
    编码方式：
    - 个体（染色体）：完整的时刻表
    - 基因：每个航班的时间偏移量（分钟）
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.3,
                 elite_size: int = 2,
                 max_offset: int = 60,
                 n_runways: int = 5):
        """
        初始化遗传算法参数
        
        Args:
            population_size: 种群大小
            generations: 迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_size: 精英保留数量            max_offset: 最大时间偏移量（分钟）
            n_runways: 机场跑道数量（默认5条）
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.max_offset = max_offset
        self.n_runways = n_runways
        
        self.objective_func = ObjectiveFunction()
        self.constraints = RunwayConstraints()
    
    def _create_individual(self, flights: List[Dict], near_zero: bool = False) -> np.ndarray:
        """
        创建一个个体（时间偏移量）
        
        Args:
            flights: 航班列表
            near_zero: True时从0附近小幅扰动出发，False时完全随机
        
        Returns:
            时间偏移量数组
        """
        if near_zero:
            # 从准点出发，加小幅随机扰动（±5分钟），让算法从接近准点的状态开始
            return np.random.uniform(-5, 5, len(flights))
        else:
            return np.random.uniform(-self.max_offset/2, self.max_offset, len(flights))
    
    def _decode_individual(self, individual: np.ndarray, 
                          flights: List[Dict]) -> List[Dict]:
        """
        解码个体为实际时刻表
        
        Args:
            individual: 时间偏移量数组
            flights: 原始航班列表
        
        Returns:
            调整后的时刻表
        """
        schedule = []
        for i, flight in enumerate(flights):
            offset_minutes = individual[i]
            scheduled_time = flight['planned_time'] + timedelta(
                minutes=offset_minutes
            )
            schedule.append({
                'flight_id': flight.get('flight_id', i),
                'planned_time': flight['planned_time'],
                'scheduled_time': scheduled_time,
                'operation': flight.get('operation', 'departure')
            })
        
        # 按时间排序并调整以满足约束
        schedule = self._adjust_for_constraints(schedule)
        return schedule
    
    def _adjust_for_constraints(self, schedule: List[Dict]) -> List[Dict]:
        """
        调整时刻表以满足安全间隔约束（支持多跑道并行）
        
        Args:
            schedule: 原始时刻表
        
        Returns:
            调整后的时刻表
        """
        sorted_schedule = sorted(schedule, key=lambda x: x['scheduled_time'])
        adjusted = []
        
        # 维护每个跑道的最后使用时间
        runway_last_times = [None] * self.n_runways
        
        for flight in sorted_schedule:
            desired_time = flight['scheduled_time']
            planned_time = flight['planned_time']
            
            # 找到最早可用的跑道
            best_runway = None
            earliest_available_time = None
            
            for runway_id in range(self.n_runways):
                if runway_last_times[runway_id] is None:
                    available_time = desired_time
                else:
                    available_time = max(
                        desired_time,
                        runway_last_times[runway_id] + timedelta(
                            minutes=self.constraints.min_interval
                        )
                    )
                
                if earliest_available_time is None or available_time < earliest_available_time:
                    earliest_available_time = available_time
                    best_runway = runway_id
            
            # 不允许提前：调度时间不能早于计划时间
            earliest_available_time = max(earliest_available_time, planned_time)
            # 安排到最佳跑道
            flight['scheduled_time'] = earliest_available_time
            flight['runway'] = best_runway + 1  # 跑道编号从1开始
            runway_last_times[best_runway] = earliest_available_time
            
            adjusted.append(flight)
        
        return adjusted
    
    def _calculate_fitness(self, individual: np.ndarray, 
                          flights: List[Dict]) -> float:
        """
        计算个体适应度（越小越好，所以返回负的惩罚值）
        
        Args:
            individual: 个体
            flights: 航班列表
        
        Returns:
            适应度值
        """
        schedule = self._decode_individual(individual, flights)
        penalty = self.objective_func.calculate_total_penalty(schedule)
        return -penalty
    
    def _selection(self, population: List[np.ndarray], 
                   fitness_scores: List[float]) -> np.ndarray:
        """
        轮盘赌选择
        
        Args:
            population: 种群
            fitness_scores: 适应度列表
        
        Returns:
            选中的个体
        """
        # 转换为正值概率
        min_fitness = min(fitness_scores)
        adjusted_fitness = [f - min_fitness + 1 for f in fitness_scores]
        total_fitness = sum(adjusted_fitness)
        
        probabilities = [f / total_fitness for f in adjusted_fitness]
        selected_idx = np.random.choice(len(population), p=probabilities)
        return population[selected_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, 
                   parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        单点交叉
        
        Args:
            parent1: 父代1
            parent2: 父代2
        
        Returns:
            两个子代
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        变异操作：微调 + 随机重置，保持种群多样性
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                if random.random() < 0.2:
                    # 20% 概率：随机重置，跳出局部最优
                    mutated[i] = np.random.uniform(-self.max_offset/2, self.max_offset)
                else:
                    # 80% 概率：微调
                    mutated[i] += np.random.uniform(-15, 15)
                    mutated[i] = np.clip(mutated[i], -self.max_offset/2, self.max_offset)
        
        # 额外的变异：交换相邻基因
        if random.random() < self.mutation_rate and len(mutated) > 1:
            idx = random.randint(0, len(mutated) - 2)
            mutated[idx], mutated[idx + 1] = mutated[idx + 1], mutated[idx]
        
        return mutated
    
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
    
    def optimize(self, flights: List[Dict], progress_callback=None) -> Dict:
        """
        执行遗传算法优化

        Args:
            flights: 航班列表
            progress_callback: 可选回调 fn(generation, total, best_penalty)
        """
        # 初始化种群：80% 从准点附近出发，20% 完全随机（保持多样性）
        n_near_zero = int(self.population_size * 0.8)
        population = (
            [self._create_individual(flights, near_zero=True) for _ in range(n_near_zero)] +
            [self._create_individual(flights, near_zero=False) for _ in range(self.population_size - n_near_zero)]
        )
        
        best_individual = None
        best_fitness = float('-inf')
        fitness_history = []
        
        print(f"  Population size: {self.population_size}, Generations: {self.generations}")
        
        for generation in range(self.generations):
            # 计算适应度
            fitness_scores = [self._calculate_fitness(ind, flights) 
                            for ind in population]
            
            # 记录最佳个体
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
            
            fitness_history.append(-best_fitness)  # 记录惩罚值
            
            # 精英保留
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elites = [population[i].copy() for i in elite_indices]
            
            # 生成新种群
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
                
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            # 输出进度
            if (generation + 1) % 10 == 0 or generation == 0:
                print(f"  Generation {generation + 1}/{self.generations} - Best penalty: {-best_fitness:.2f}")
                if progress_callback:
                    progress_callback(generation + 1, self.generations, -best_fitness)
        
        # 返回最优解
        best_schedule = self._decode_individual(best_individual, flights)
        
        return {
            'algorithm': 'Genetic Algorithm',
            'schedule': best_schedule,
            'penalty': -best_fitness,
            'fitness_history': fitness_history,
            'generations': self.generations
        }
