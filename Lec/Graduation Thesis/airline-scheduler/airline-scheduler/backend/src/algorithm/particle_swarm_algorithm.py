"""
粒子群算法（鸟群算法）模块
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


class ParticleSwarmAlgorithm:
    """
    粒子群算法类：通过粒子群体协作寻找最优时刻表
    
    编码方式：
    - 粒子位置：每个航班的时间偏移量（分钟）
    - 粒子速度：时刻表的调整方向和幅度
    """
    
    def __init__(self,
                 n_particles: int = 30,
                 n_iterations: int = 100,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 max_velocity: float = 20.0,
                 max_offset: float = 60.0,
                 n_runways: int = 5):
        """
        初始化粒子群算法参数
        
        Args:
            n_particles: 粒子数量
            n_iterations: 迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            max_velocity: 最大速度（分钟）
            max_offset: 最大时间偏移量（分钟）
            n_runways: 机场跑道数量（默认5条）
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_velocity = max_velocity
        self.max_offset = max_offset
        self.n_runways = n_runways
        
        self.objective_func = ObjectiveFunction()
        self.constraints = RunwayConstraints()
    
    def _initialize_particles(self, n_flights: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        初始化粒子位置和速度
        
        Args:
            n_flights: 航班数量
        
        Returns:
            (位置矩阵, 速度矩阵)
        """
        # 位置：时间偏移量，初始化在 [-max_offset/2, max_offset] 范围内
        positions = np.random.uniform(-self.max_offset/2, self.max_offset,
                                     (self.n_particles, n_flights))
        
        # 速度：初始化在 [-max_velocity, max_velocity] 范围内
        velocities = np.random.uniform(-self.max_velocity, self.max_velocity,
                                      (self.n_particles, n_flights))
        
        return positions, velocities
    
    def _decode_position(self, position: np.ndarray, 
                        flights: List[Dict]) -> List[Dict]:
        """
        将粒子位置解码为时刻表
        
        Args:
            position: 粒子位置（时间偏移量数组）
            flights: 航班列表
        
        Returns:
            时刻表
        """
        schedule = []
        for i, flight in enumerate(flights):
            offset_minutes = position[i]
            scheduled_time = flight['planned_time'] + timedelta(
                minutes=offset_minutes
            )
            schedule.append({
                'flight_id': flight.get('flight_id', i),
                'planned_time': flight['planned_time'],
                'scheduled_time': scheduled_time,
                'operation': flight.get('operation', 'departure')
            })
        
        # 调整以满足约束
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
            
            # 找到最早可用的跑道
            best_runway = None
            earliest_available_time = None
            
            for runway_id in range(self.n_runways):
                if runway_last_times[runway_id] is None:
                    # 跑道未使用，可以使用期望时间
                    available_time = desired_time
                else:
                    # 跑道已使用，需要等待安全间隔
                    available_time = max(
                        desired_time,
                        runway_last_times[runway_id] + timedelta(
                            minutes=self.constraints.min_interval
                        )
                    )
                
                # 选择最早可用的跑道
                if earliest_available_time is None or available_time < earliest_available_time:
                    earliest_available_time = available_time
                    best_runway = runway_id
            
            # 安排到最佳跑道
            flight['scheduled_time'] = earliest_available_time
            flight['runway'] = best_runway + 1  # 跑道编号从1开始
            runway_last_times[best_runway] = earliest_available_time
            
            adjusted.append(flight)
        
        return adjusted
    
    def _calculate_fitness(self, position: np.ndarray, 
                          flights: List[Dict]) -> float:
        """
        计算粒子适应度（惩罚值，越小越好）
        
        Args:
            position: 粒子位置
            flights: 航班列表
        
        Returns:
            惩罚值
        """
        schedule = self._decode_position(position, flights)
        penalty = self.objective_func.calculate_total_penalty(schedule)
        return penalty
    
    def _update_velocity(self, velocity: np.ndarray, 
                        position: np.ndarray,
                        personal_best: np.ndarray,
                        global_best: np.ndarray) -> np.ndarray:
        """
        更新粒子速度
        
        Args:
            velocity: 当前速度
            position: 当前位置
            personal_best: 个体历史最佳位置
            global_best: 全局历史最佳位置
        
        Returns:
            新速度
        """
        r1 = np.random.random(len(velocity))
        r2 = np.random.random(len(velocity))
        
        # PSO速度更新公式
        cognitive = self.c1 * r1 * (personal_best - position)
        social = self.c2 * r2 * (global_best - position)
        new_velocity = self.w * velocity + cognitive + social
        
        # 限制速度范围
        new_velocity = np.clip(new_velocity, -self.max_velocity, 
                              self.max_velocity)
        
        return new_velocity
    
    def _update_position(self, position: np.ndarray, 
                        velocity: np.ndarray) -> np.ndarray:
        """
        更新粒子位置
        
        Args:
            position: 当前位置
            velocity: 当前速度
        
        Returns:
            新位置
        """
        new_position = position + velocity
        
        # 限制位置范围
        new_position = np.clip(new_position, -self.max_offset/2, 
                              self.max_offset)
        
        return new_position
    
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
        执行粒子群算法优化
        
        Args:
            flights: 航班列表
        
        Returns:
            优化结果字典
        """
        n_flights = len(flights)
        
        print(f"  粒子数量: {self.n_particles}, 迭代次数: {self.n_iterations}")
        
        # 初始化粒子
        positions, velocities = self._initialize_particles(n_flights)
        
        # 初始化个体最佳和全局最佳
        personal_best_positions = positions.copy()
        personal_best_fitness = np.array([
            self._calculate_fitness(pos, flights) for pos in positions
        ])
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        fitness_history = []
        
        # 迭代优化
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # 更新速度和位置
                velocities[i] = self._update_velocity(
                    velocities[i],
                    positions[i],
                    personal_best_positions[i],
                    global_best_position
                )
                
                positions[i] = self._update_position(positions[i], velocities[i])
                
                # 计算适应度
                fitness = self._calculate_fitness(positions[i], flights)
                
                # 更新个体最佳
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = positions[i].copy()
                
                # 更新全局最佳
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = positions[i].copy()
            
            fitness_history.append(global_best_fitness)
            
            # 动态调整惯性权重（线性递减）
            self.w = 0.9 - (0.9 - 0.4) * iteration / self.n_iterations
            
            # 输出进度
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"  迭代 {iteration + 1}/{self.n_iterations} - 最佳惩罚值: {global_best_fitness:.2f}")
        
        # 解码最优解
        best_schedule = self._decode_position(global_best_position, flights)
        
        return {
            'algorithm': 'Particle Swarm Algorithm',
            'schedule': best_schedule,
            'penalty': global_best_fitness,
            'fitness_history': fitness_history,
            'iterations': self.n_iterations
        }
