"""
测试数据预处理模块
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import preprocess_flight_data

# ==================== 配置参数 ====================
# 修改这些参数来配置预处理

CSV_PATH = "../dataset/dataset/processed/Brazilian-flights.csv"
AIRPORT_CODE = "SBGR"  # 圣保罗瓜鲁柳斯国际机场
N_RUNWAYS = 5
SAFETY_INTERVAL = 3
START_DATE = "2016-01-01"
END_DATE = "2016-01-31"

# ==================================================

def test_preprocessor():
    """测试预处理器功能"""
    print("=" * 60)
    print("开始测试数据预处理模块")
    print("=" * 60)
    
    print("\n配置参数:")
    print(f"  数据文件: {CSV_PATH}")
    print(f"  机场代码: {AIRPORT_CODE}")
    print(f"  跑道数量: {N_RUNWAYS}")
    print(f"  安全间隔: {SAFETY_INTERVAL} 分钟")
    print(f"  日期范围: {START_DATE} 至 {END_DATE}")
    
    try:
        # 执行预处理
        df_clean, algorithm_input = preprocess_flight_data(
            csv_path=CSV_PATH,
            airport_code=AIRPORT_CODE,
            n_runways=N_RUNWAYS,
            safety_interval=SAFETY_INTERVAL,
            start_date=START_DATE,
            end_date=END_DATE,
            log_dir="../src/output"
        )
        
        print("\n" + "=" * 60)
        print("预处理完成！")
        print("=" * 60)
        
        # 显示结果摘要
        print(f"\n清洗后数据规模: {df_clean.shape[0]} 行 × {df_clean.shape[1]} 列")
        print(f"\n前5行数据:")
        print(df_clean.head())
        
        print(f"\n算法输入摘要:")
        print(f"  机场: {algorithm_input['airport']}")
        print(f"  跑道数: {algorithm_input['n_runways']}")
        print(f"  安全间隔: {algorithm_input['safety_interval_minutes']} 分钟")
        print(f"  总事件数: {algorithm_input['total_events']}")
        print(f"  起飞事件: {algorithm_input['departure_events']}")
        print(f"  降落事件: {algorithm_input['arrival_events']}")
        
        print(f"\n前5个事件:")
        for i, event in enumerate(algorithm_input['events'][:5]):
            print(f"  {i+1}. {event['event_type']}: {event['scheduled_time']}")
        
        print("\n测试成功！日志文件已保存到 ../src/output/preprocessed 目录")
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocessor()
