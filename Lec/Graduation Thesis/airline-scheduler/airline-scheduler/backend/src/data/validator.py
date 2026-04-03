"""
数据集深度验证模块
"""

import pandas as pd
from typing import Tuple

REQUIRED_COLUMNS = [
    'Flight.No', 'Airport.From', 'Airport.To',
    'Scheduled.Departure', 'Scheduled.Arrival',
    'Departure', 'Arrival',
    'Distance.In.Meters',
    'Longitude.From', 'Latitude.From',
    'Longitude.To', 'Latitude.To'
]


def validate(df: pd.DataFrame, airport_code: str) -> Tuple[bool, str]:
    """
    对数据集进行深度验证

    Returns:
        (passed, error_message)  passed=True 表示验证通过
    """
    # 1. 列头检查
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"缺少必要列：{', '.join(missing)}"

    # 2. 行数检查
    if len(df) == 0:
        return False, "数据集为空"

    # 3. 目标机场是否存在
    airport_rows = df[(df['Airport.From'] == airport_code) | (df['Airport.To'] == airport_code)]
    if len(airport_rows) == 0:
        return False, f"数据集中未找到目标机场 {airport_code} 的航班记录"

    # 4. 时间格式检查
    for col in ['Scheduled.Departure', 'Scheduled.Arrival']:
        parsed = pd.to_datetime(df[col], errors='coerce')
        bad = parsed.isna().sum()
        if bad > len(df) * 0.5:
            return False, f"列 {col} 中超过 50% 的值无法解析为时间格式"

    # 5. 经纬度范围检查
    for lon_col in ['Longitude.From', 'Longitude.To']:
        col = pd.to_numeric(df[lon_col], errors='coerce')
        if col.notna().any():
            if not (col.dropna().between(-180, 180).all()):
                return False, f"列 {lon_col} 存在超出 [-180, 180] 范围的值"

    for lat_col in ['Latitude.From', 'Latitude.To']:
        col = pd.to_numeric(df[lat_col], errors='coerce')
        if col.notna().any():
            if not (col.dropna().between(-90, 90).all()):
                return False, f"列 {lat_col} 存在超出 [-90, 90] 范围的值"

    return True, ""
