"""
约束条件模块
用于检查和验证跑道安全间隔等约束
"""

from datetime import datetime, timedelta
from typing import List, Dict


class RunwayConstraints:
    """
    跑道约束类：管理跑道使用的安全间隔
    """
    
    def __init__(self, min_interval_minutes: float = 3.0):
        """
        初始化约束条件
        
        Args:
            min_interval_minutes: 最小安全间隔（分钟），默认3分钟
        """
        self.min_interval = min_interval_minutes
    
    def check_interval(self, time1: datetime, time2: datetime) -> bool:
        """
        检查两个时间是否满足最小间隔要求
        
        Args:
            time1: 第一个时间
            time2: 第二个时间
        
        Returns:
            True 如果满足间隔要求，False 否则
        """
        interval_minutes = abs((time2 - time1).total_seconds() / 60)
        return interval_minutes >= self.min_interval
    
    def validate_schedule(self, schedule: List[Dict]) -> tuple[bool, List[str]]:
        """
        验证完整时刻表是否满足所有约束
        
        Args:
            schedule: 时刻表列表，每个元素包含：
                - 'scheduled_time': 排班时间
                - 'flight_id': 航班标识
                - 'operation': 操作类型 ('departure' 或 'arrival')
        
        Returns:
            (是否有效, 违规信息列表)
        """
        if not schedule:
            return True, []
        
        # 按时间排序
        sorted_schedule = sorted(schedule, key=lambda x: x['scheduled_time'])
        
        violations = []
        
        # 检查相邻航班间隔
        for i in range(len(sorted_schedule) - 1):
            current = sorted_schedule[i]
            next_flight = sorted_schedule[i + 1]
            
            if not self.check_interval(current['scheduled_time'], 
                                      next_flight['scheduled_time']):
                interval = (next_flight['scheduled_time'] - 
                           current['scheduled_time']).total_seconds() / 60
                violations.append(
                    f"航班 {current.get('flight_id', i)} 和 "
                    f"{next_flight.get('flight_id', i+1)} 间隔不足: "
                    f"{interval:.2f}分钟 < {self.min_interval}分钟"
                )
        
        return len(violations) == 0, violations
    
    def get_earliest_valid_time(self, reference_time: datetime, 
                               existing_times: List[datetime]) -> datetime:
        """
        获取满足约束的最早可用时间
        
        Args:
            reference_time: 参考时间（期望时间）
            existing_times: 已占用的时间列表
        
        Returns:
            满足约束的最早时间
        """
        if not existing_times:
            return reference_time
        
        # 排序已有时间
        sorted_times = sorted(existing_times)
        
        candidate_time = reference_time
        
        # 检查是否与任何已有时间冲突
        while True:
            conflict = False
            for existing_time in sorted_times:
                if not self.check_interval(candidate_time, existing_time):
                    # 有冲突，调整到该时间之后
                    if candidate_time <= existing_time:
                        candidate_time = existing_time + timedelta(
                            minutes=self.min_interval
                        )
                        conflict = True
                        break
            
            if not conflict:
                break
        
        return candidate_time
