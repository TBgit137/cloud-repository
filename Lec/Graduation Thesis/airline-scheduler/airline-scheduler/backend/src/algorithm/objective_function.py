"""
目标函数模块
用于计算航班调度的延误惩罚，目标是最小化总延误
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Union


class ObjectiveFunction:
    """
    目标函数类：计算航班调度的延误惩罚
    
    惩罚规则：
    - 排班时间早于或等于预计时间：无惩罚（0分）
    - 排班时间晚于预计时间：根据延误时长进行非线性惩罚
    """
    
    def __init__(self, penalty_type: str = 'piecewise'):
        """
        初始化目标函数
        
        Args:
            penalty_type: 惩罚函数类型
                - 'piecewise': 分段函数（默认）
                - 'exponential': 指数函数
        
        惩罚规则（提前和延误完全对称，偏差越大惩罚越重，准点惩罚为0）：
            - 偏差 0 分钟：准点，无惩罚
            - 偏差 0-10 分钟：线性惩罚（1倍）
            - 偏差 10-30 分钟：加速惩罚（3倍）
            - 偏差 30-60 分钟：重度惩罚（8倍）
            - 偏差 60 分钟以上：极重惩罚（15倍）
        """
        self.penalty_type = penalty_type

    def _piecewise_penalty(self, delay_minutes: float) -> float:
        """
        分段惩罚函数，提前和延误对称处理
        
        Args:
            delay_minutes: 偏差时间（分钟），负值=提前，正值=延误
        
        Returns:
            惩罚分数（始终非负，准点时为0）
        """
        if delay_minutes == 0:
            return 0.0
        
        deviation = abs(delay_minutes)
        # 提前惩罚是延误惩罚的3倍，强烈驱动算法不提前
        multiplier = 3.0 if delay_minutes < 0 else 1.0
        
        if deviation <= 10:
            base = deviation
        elif deviation <= 30:
            base = 10 + 3 * ((deviation - 10) ** 1.5)
        elif deviation <= 60:
            base = 10 + 3 * (20 ** 1.5) + 8 * ((deviation - 30) ** 1.8)
        else:
            base = 10 + 3 * (20 ** 1.5) + 8 * (30 ** 1.8) + 15 * ((deviation - 60) ** 2.0)
        
        return multiplier * base    
    def _exponential_penalty(self, delay_minutes: float) -> float:
        """
        指数惩罚函数，提前和延误对称处理
        
        Args:
            delay_minutes: 偏差时间（分钟），负值=提前，正值=延误
        
        Returns:
            惩罚分数（始终非负，准点时为0）
        """
        deviation = abs(delay_minutes)
        if deviation == 0:
            return 0.0
        return 0.05 * (deviation ** 2.2)
    
    def calculate_delay_penalty(self, scheduled_time: datetime, 
                               planned_time: datetime) -> float:
        """
        计算单个航班的延误惩罚
        
        Args:
            scheduled_time: 排班时间（算法分配的时间）
            planned_time: 预计时间（原计划时间）
        
        Returns:
            延误惩罚分数（非负数）
        """
        # 计算延误时间（分钟）
        delay_minutes = (scheduled_time - planned_time).total_seconds() / 60
        
        # 根据选择的惩罚类型计算惩罚
        if self.penalty_type == 'piecewise':
            return self._piecewise_penalty(delay_minutes)
        else:
            return self._exponential_penalty(delay_minutes)
    
    def calculate_total_penalty(self, flights: List[Dict]) -> float:
        """
        计算所有航班的总延误惩罚
        
        Args:
            flights: 航班列表，每个航班包含：
                - 'scheduled_time': 排班时间
                - 'planned_time': 预计时间
                - 'flight_type': 航班类型（'departure' 或 'arrival'）
        
        Returns:
            总延误惩罚分数
        """
        total_penalty = 0.0
        
        for flight in flights:
            penalty = self.calculate_delay_penalty(
                flight['scheduled_time'],
                flight['planned_time']
            )
            total_penalty += penalty
        
        return total_penalty
    
    def evaluate_schedule(self, departures: List[Dict], 
                         arrivals: List[Dict]) -> Dict[str, float]:
        """
        评估完整的起降时刻表
        
        Args:
            departures: 起飞航班列表
            arrivals: 降落航班列表
        
        Returns:
            评估结果字典，包含：
                - 'departure_penalty': 起飞延误惩罚
                - 'arrival_penalty': 降落延误惩罚
                - 'total_penalty': 总惩罚
        """
        departure_penalty = self.calculate_total_penalty(departures)
        arrival_penalty = self.calculate_total_penalty(arrivals)
        
        return {
            'departure_penalty': departure_penalty,
            'arrival_penalty': arrival_penalty,
            'total_penalty': departure_penalty + arrival_penalty
        }
    
    def get_penalty_curve(self, max_delay_minutes: int = 120) -> np.ndarray:
        """
        获取惩罚曲线数据（用于可视化和分析）
        
        Args:
            max_delay_minutes: 最大延误时间（分钟）
        
        Returns:
            二维数组 [[延误时间, 惩罚分数], ...]
        """
        delays = np.arange(0, max_delay_minutes + 1)
        penalties = np.zeros(len(delays))
        
        for i, delay in enumerate(delays):
            if self.penalty_type == 'piecewise':
                penalties[i] = self._piecewise_penalty(float(delay))
            else:
                penalties[i] = self._exponential_penalty(float(delay))
        
        return np.column_stack((delays, penalties))
    
    def get_penalty_statistics(self, delays_minutes: List[float]) -> Dict[str, float]:
        """
        获取延误惩罚的统计信息
        
        Args:
            delays_minutes: 延误时间列表（分钟）
        
        Returns:
            统计信息字典
        """
        if not delays_minutes:
            return {
                'total_flights': 0,
                'avg_delay': 0.0,
                'max_delay': 0.0,
                'total_penalty': 0.0,
                'avg_penalty': 0.0
            }
        
        penalties = []
        for delay in delays_minutes:
            if self.penalty_type == 'piecewise':
                penalties.append(self._piecewise_penalty(delay))
            else:
                penalties.append(self._exponential_penalty(delay))
        
        return {
            'total_flights': len(delays_minutes),
            'avg_delay': np.mean(delays_minutes),
            'max_delay': np.max(delays_minutes),
            'min_delay': np.min(delays_minutes),
            'total_penalty': np.sum(penalties),
            'avg_penalty': np.mean(penalties),
            'max_penalty': np.max(penalties)
        }
