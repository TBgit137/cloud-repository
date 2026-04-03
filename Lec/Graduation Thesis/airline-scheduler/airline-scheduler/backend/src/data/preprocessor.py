"""
数据预处理模块
负责清洗和准备航班数据，为调度算法提供输入
"""

import pandas as pd
import numpy as np
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz
import os
import json
from typing import Dict, List, Tuple, Optional


class FlightDataPreprocessor:
    """航班数据预处理器"""
    
    def __init__(self, log_dir: str = "../output"):
        """
        初始化预处理器
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        self.tf = TimezoneFinder()
        self.processing_log = {
            'timestamp': datetime.now().isoformat(),
            'steps': [],
            'statistics': {}
        }
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
    
    def process(self, 
                df: pd.DataFrame,
                airport_code: str,
                n_runways: int,
                safety_interval: int,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        完整的数据预处理流程
        
        Args:
            df: 原始航班数据
            airport_code: 指定机场代码
            n_runways: 可同时使用的跑道数
            safety_interval: 最短跑道安全间隔（分钟）
            start_date: 开始日期 (yyyy-mm-dd)，None表示不限
            end_date: 结束日期 (yyyy-mm-dd)，None表示不限
            max_flights: 最大航班数量，None表示不限
            
        Returns:
            (清洗后的数据框, 算法输入数据字典)
        """
        self._log_step("开始数据预处理", {
            'original_rows': len(df),
            'airport': airport_code,
            'n_runways': n_runways,
            'safety_interval': safety_interval,
            'date_range': f"{start_date} to {end_date}"
        })
        
        # 1. 筛选指定机场和时间段的航班
        df_filtered = self._filter_by_airport_and_date(
            df, airport_code, start_date, end_date
        )
        
        # 2. 删除缺失值和重复值
        df_clean = self._remove_missing_and_duplicates(df_filtered)
        
        # 3. 删除时间逻辑错误的航班
        df_valid = self._remove_time_logic_errors(df_clean)
        
        # 4. 转换为UTC时间
        df_utc = self._convert_to_utc(df_valid)
        
        # 5. 验证航行时间，删除极端异常
        df_verified = self._verify_flight_time(df_utc)
        
        # 6. 添加唯一标识
        df_final = self._add_unique_id(df_verified)
        
        # 7. 准备算法输入数据
        algorithm_input = self._prepare_algorithm_input(
            df_final, airport_code, n_runways, safety_interval
        )
        
        # 8. 保存结果和日志
        self._save_results(df_final, algorithm_input)
        
        self._log_step("数据预处理完成", {
            'final_rows': len(df_final),
            'removed_rows': len(df) - len(df_final),
            'removal_rate': f"{(len(df) - len(df_final)) / len(df) * 100:.2f}%"
        })
        
        return df_final, algorithm_input, df_filtered
    
    def _filter_by_airport_and_date(self, 
                                     df: pd.DataFrame,
                                     airport_code: str,
                                     start_date: Optional[str],
                                     end_date: Optional[str]) -> pd.DataFrame:
        """筛选指定机场和时间段的航班"""
        df_copy = df.copy()
        original_count = len(df_copy)
        
        # 筛选机场（起飞或降落机场为指定机场）
        mask_airport = (df_copy['Airport.From'] == airport_code) | \
                      (df_copy['Airport.To'] == airport_code)
        df_copy = df_copy[mask_airport].copy()
        
        airport_filtered = len(df_copy)
        
        # 筛选时间段
        if start_date or end_date:
            # 确保时间列是datetime格式
            df_copy['Scheduled.Departure'] = pd.to_datetime(
                df_copy['Scheduled.Departure'], errors='coerce'
            )
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df_copy = df_copy[df_copy['Scheduled.Departure'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                df_copy = df_copy[df_copy['Scheduled.Departure'] < end_dt]
        
        date_filtered = len(df_copy)
        
        self._log_step("筛选机场和时间段", {
            'original': original_count,
            'after_airport_filter': airport_filtered,
            'after_date_filter': date_filtered,
            'removed': original_count - date_filtered
        })
        
        return df_copy.reset_index(drop=True)
    
    def _remove_missing_and_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除缺失值和重复值"""
        df_copy = df.copy()
        original_count = len(df_copy)
        
        # 删除计划起飞/降落时间缺失的行
        required_cols = ['Scheduled.Departure', 'Scheduled.Arrival']
        df_copy = df_copy.dropna(subset=required_cols)
        after_missing = len(df_copy)
        
        # 删除重复行
        df_copy = df_copy.drop_duplicates()
        after_duplicates = len(df_copy)
        
        self._log_step("删除缺失值和重复值", {
            'original': original_count,
            'missing_removed': original_count - after_missing,
            'duplicates_removed': after_missing - after_duplicates,
            'remaining': after_duplicates
        })
        
        return df_copy.reset_index(drop=True)
    
    def _remove_time_logic_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除时间逻辑错误的航班"""
        df_copy = df.copy()
        original_count = len(df_copy)
        
        # 确保时间列是datetime格式
        time_cols = ['Scheduled.Departure', 'Scheduled.Arrival', 
                     'Departure', 'Arrival']
        for col in time_cols:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        
        # 删除计划起飞时间晚于计划降落时间的
        mask_scheduled = df_copy['Scheduled.Departure'] <= df_copy['Scheduled.Arrival']
        df_copy = df_copy[mask_scheduled]
        after_scheduled = len(df_copy)
        
        # 删除实际起飞时间晚于实际降落时间的（如果有实际时间）
        has_actual = df_copy['Departure'].notna() & df_copy['Arrival'].notna()
        mask_actual = ~has_actual | (df_copy['Departure'] <= df_copy['Arrival'])
        df_copy = df_copy[mask_actual]
        after_actual = len(df_copy)
        
        self._log_step("删除时间逻辑错误", {
            'original': original_count,
            'scheduled_time_errors': original_count - after_scheduled,
            'actual_time_errors': after_scheduled - after_actual,
            'remaining': after_actual
        })
        
        return df_copy.reset_index(drop=True)
    
    def _get_timezone_from_coords(self, lon: float, lat: float) -> str:
        """根据经纬度获取时区"""
        try:
            tz_name = self.tf.timezone_at(lng=lon, lat=lat)
            return tz_name if tz_name else 'UTC'
        except:
            return 'UTC'
    
    def _convert_local_to_utc(self, local_time: pd.Timestamp, 
                              tz_name: str) -> pd.Timestamp:
        """将本地时间转换为UTC"""
        if pd.isna(local_time) or tz_name is None:
            return pd.NaT
        try:
            tz = pytz.timezone(tz_name)
            local_dt = tz.localize(local_time.replace(tzinfo=None))
            utc_dt = local_dt.astimezone(pytz.UTC)
            return utc_dt.replace(tzinfo=None)
        except:
            return pd.NaT
    
    def _convert_to_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换所有时间为UTC"""
        df_copy = df.copy()
        original_count = len(df_copy)
        
        # 创建机场时区映射
        airports_from = df_copy[['Airport.From', 'Longitude.From', 'Latitude.From']].drop_duplicates()
        airports_to = df_copy[['Airport.To', 'Longitude.To', 'Latitude.To']].drop_duplicates()
        
        airport_tz = {}
        
        for _, row in airports_from.iterrows():
            airport = row['Airport.From']
            if airport not in airport_tz:
                tz = self._get_timezone_from_coords(
                    row['Longitude.From'], row['Latitude.From']
                )
                airport_tz[airport] = tz
        
        for _, row in airports_to.iterrows():
            airport = row['Airport.To']
            if airport not in airport_tz:
                tz = self._get_timezone_from_coords(
                    row['Longitude.To'], row['Latitude.To']
                )
                airport_tz[airport] = tz
        
        # 添加时区信息
        df_copy['Departure_TZ'] = df_copy['Airport.From'].map(airport_tz)
        df_copy['Arrival_TZ'] = df_copy['Airport.To'].map(airport_tz)
        
        # 转换时间为UTC
        df_copy['Scheduled.Departure.UTC'] = df_copy.apply(
            lambda row: self._convert_local_to_utc(
                row['Scheduled.Departure'], row['Departure_TZ']
            ), axis=1
        )
        
        df_copy['Scheduled.Arrival.UTC'] = df_copy.apply(
            lambda row: self._convert_local_to_utc(
                row['Scheduled.Arrival'], row['Arrival_TZ']
            ), axis=1
        )
        
        # 转换实际时间（如果存在）
        if df_copy['Departure'].notna().any():
            df_copy['Departure.UTC'] = df_copy.apply(
                lambda row: self._convert_local_to_utc(
                    row['Departure'], row['Departure_TZ']
                ) if pd.notna(row['Departure']) else pd.NaT, axis=1
            )
        
        if df_copy['Arrival'].notna().any():
            df_copy['Arrival.UTC'] = df_copy.apply(
                lambda row: self._convert_local_to_utc(
                    row['Arrival'], row['Arrival_TZ']
                ) if pd.notna(row['Arrival']) else pd.NaT, axis=1
            )
        
        # 删除UTC转换失败的行
        df_copy = df_copy.dropna(subset=['Scheduled.Departure.UTC', 'Scheduled.Arrival.UTC'])
        after_utc = len(df_copy)
        
        self._log_step("转换为UTC时间", {
            'original': original_count,
            'timezone_mapping_created': len(airport_tz),
            'utc_conversion_failed': original_count - after_utc,
            'remaining': after_utc
        })
        
        return df_copy.reset_index(drop=True)
    
    def _verify_flight_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证航行时间，删除极端异常"""
        df_copy = df.copy()
        original_count = len(df_copy)
        
        # 如果数据为空，直接返回
        if len(df_copy) == 0:
            self._log_step("验证航行时间", {
                'original': 0,
                'too_slow_removed': 0,
                'too_fast_removed': 0,
                'total_removed': 0,
                'remaining': 0,
                'speed_range': '50-300 m/s'
            })
            return df_copy
        
        # 计算航行时间（秒）
        df_copy['Flight_Duration_Seconds'] = (
            df_copy['Scheduled.Arrival.UTC'] - df_copy['Scheduled.Departure.UTC']
        ).dt.total_seconds()
        
        # 计算平均速度 (m/s)
        df_copy['Average_Speed_mps'] = df_copy['Distance.In.Meters'] / df_copy['Flight_Duration_Seconds']
        
        # 定义合理速度范围
        # 民用航班最高速度约 300 m/s (约1080 km/h)
        # 最低速度 50 m/s (约180 km/h，起飞速度以下视为异常)
        MIN_SPEED = 50  # m/s
        MAX_SPEED = 300  # m/s
        
        # 筛选合理速度的航班
        mask_speed = (df_copy['Average_Speed_mps'] >= MIN_SPEED) & \
                     (df_copy['Average_Speed_mps'] <= MAX_SPEED)
        
        df_valid = df_copy[mask_speed].copy()
        after_verification = len(df_valid)
        
        # 统计异常情况
        too_slow = int((df_copy['Average_Speed_mps'] < MIN_SPEED).sum())
        too_fast = int((df_copy['Average_Speed_mps'] > MAX_SPEED).sum())
        
        self._log_step("验证航行时间", {
            'original': original_count,
            'too_slow_removed': too_slow,
            'too_fast_removed': too_fast,
            'total_removed': original_count - after_verification,
            'remaining': after_verification,
            'speed_range': f"{MIN_SPEED}-{MAX_SPEED} m/s"
        })
        
        # 删除临时计算列
        df_valid = df_valid.drop(columns=['Flight_Duration_Seconds', 'Average_Speed_mps'])
        
        return df_valid.reset_index(drop=True)
    
    def _add_unique_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加唯一标识"""
        df_copy = df.copy()
        
        # 使用顺序整数ID
        df_copy.insert(0, 'Flight_ID', range(1, len(df_copy) + 1))
        
        self._log_step("添加唯一标识", {
            'total_ids_generated': len(df_copy),
            'id_range': f"1-{len(df_copy)}"
        })
        
        return df_copy
    
    def _prepare_algorithm_input(self, 
                                  df: pd.DataFrame,
                                  airport_code: str,
                                  n_runways: int,
                                  safety_interval: int) -> Dict:
        """准备算法输入数据"""
        
        # 筛选起飞航班（出发机场为指定机场）
        departures = df[df['Airport.From'] == airport_code].copy()
        
        # 筛选降落航班（到达机场为指定机场）
        arrivals = df[df['Airport.To'] == airport_code].copy()
        
        # 合并所有事件（起飞和降落都视为跑道占用事件）
        events = []
        
        for _, row in departures.iterrows():
            events.append({
                'flight_id': row['Flight_ID'],
                'event_type': 'departure',
                'scheduled_time': row['Scheduled.Departure.UTC'].isoformat(),
                'airport': airport_code
            })
        
        for _, row in arrivals.iterrows():
            events.append({
                'flight_id': row['Flight_ID'],
                'event_type': 'arrival',
                'scheduled_time': row['Scheduled.Arrival.UTC'].isoformat(),
                'airport': airport_code
            })
        
        # 按时间排序
        events.sort(key=lambda x: x['scheduled_time'])
        
        algorithm_input = {
            'airport': airport_code,
            'n_runways': n_runways,
            'safety_interval_minutes': safety_interval,
            'total_events': len(events),
            'departure_events': len(departures),
            'arrival_events': len(arrivals),
            'events': events
        }
        
        self._log_step("准备算法输入", {
            'total_events': len(events),
            'departures': len(departures),
            'arrivals': len(arrivals),
            'n_runways': n_runways,
            'safety_interval': safety_interval
        })
        
        return algorithm_input
    
    def _save_results(self, df: pd.DataFrame, algorithm_input: Dict):
        """保存处理结果和日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 预处理结果保存到 output/preprocessed/ 子目录
        preprocessed_dir = os.path.join(self.log_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        # 保存清洗后的数据
        output_file = os.path.join(preprocessed_dir, f"cleaned_data_{timestamp}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # 保存算法输入数据
        input_file = os.path.join(preprocessed_dir, f"algorithm_input_{timestamp}.json")
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(algorithm_input, f, indent=2, ensure_ascii=False)
        
        # 保存处理日志
        log_file = os.path.join(preprocessed_dir, f"processing_log_{timestamp}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.processing_log, f, indent=2, ensure_ascii=False)
        
        self._log_step("保存结果", {
            'cleaned_data_file': output_file,
            'algorithm_input_file': input_file,
            'log_file': log_file
        })
    
    def _log_step(self, step_name: str, details: Dict):
        """记录处理步骤"""
        self.processing_log['steps'].append({
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'details': details
        })
        print(f"[{step_name}] {details}")


def preprocess_flight_data(csv_path: str,
                           airport_code: str,
                           n_runways: int = 5,
                           safety_interval: int = 3,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           log_dir: str = "../output",
                           _df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    便捷函数：预处理航班数据

    Args:
        csv_path: CSV文件路径（与 _df 二选一）
        airport_code: 机场代码
        n_runways: 可同时使用的跑道数
        safety_interval: 最短跑道安全间隔（分钟）
        start_date: 开始日期 (yyyy-mm-dd)
        end_date: 结束日期 (yyyy-mm-dd)
        log_dir: 日志目录
        _df: 直接传入已读取的 DataFrame（优先于 csv_path）

    Returns:
        (清洗后的数据框, 算法输入数据字典)
    """
    # 加载数据
    if _df is not None:
        df = _df.copy()
    else:
        df = pd.read_csv(csv_path, encoding='latin1')
    
    # 创建预处理器
    preprocessor = FlightDataPreprocessor(log_dir=log_dir)
    
    # 执行预处理
    df_clean, algorithm_input, df_filtered = preprocessor.process(
        df=df,
        airport_code=airport_code,
        n_runways=n_runways,
        safety_interval=safety_interval,
        start_date=start_date,
        end_date=end_date
    )
    
    return df_clean, algorithm_input, df_filtered
