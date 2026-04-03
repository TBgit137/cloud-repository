"""
排班表评价脚本
将算法输出的排班表与清洗后数据中的实际起降时间对比，
评估算法调度是否比原始计划更接近实际情况（即总延误是否缓解）
"""

import sys
import os
import glob
import pandas as pd

# ==================== 配置参数 ====================
AIRPORT_CODE = "SBGR"
LOG_DIR = "../src/output"
# 留空则自动选最新文件
SCHEDULE_FILE = ""   # 排班表 CSV，留空自动选最新
CLEANED_FILE  = ""   # 清洗后数据 CSV，留空自动选最新
# ==================================================


def find_latest(directory: str, pattern: str) -> str:
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"在 {directory} 中找不到匹配 {pattern} 的文件")
    return max(files, key=os.path.getmtime)


def load_files(log_dir: str, schedule_file: str, cleaned_file: str):
    schedule_path = schedule_file or find_latest(
        os.path.join(log_dir, "results"), "schedule_result_*.csv"
    )
    cleaned_path = cleaned_file or find_latest(
        os.path.join(log_dir, "preprocessed"), "cleaned_data_*.csv"
    )
    print(f"Schedule file:  {os.path.basename(schedule_path)}")
    print(f"Cleaned data:   {os.path.basename(cleaned_path)}")
    return pd.read_csv(schedule_path), pd.read_csv(cleaned_path, low_memory=False)


def compute_actual_delay(cleaned: pd.DataFrame, airport: str) -> pd.DataFrame:
    """
    从清洗后数据提取每个 Flight_ID 的实际延误（分钟）
    - 目标机场起飞：Departure.UTC - Scheduled.Departure.UTC
    - 目标机场降落：Arrival.UTC   - Scheduled.Arrival.UTC
    """
    rows = []

    # 起飞事件：Airport.From == airport
    dep = cleaned[cleaned["Airport.From"] == airport].copy()
    dep["scheduled_utc"] = pd.to_datetime(dep["Scheduled.Departure.UTC"], errors="coerce")
    dep["actual_utc"]    = pd.to_datetime(dep["Departure.UTC"], errors="coerce")
    dep = dep.dropna(subset=["scheduled_utc", "actual_utc"])
    dep["actual_delay_min"] = (dep["actual_utc"] - dep["scheduled_utc"]).dt.total_seconds() / 60
    dep["event_type"] = "departure"
    rows.append(dep[["Flight_ID", "event_type", "scheduled_utc", "actual_utc", "actual_delay_min"]])

    # 降落事件：Airport.To == airport
    arr = cleaned[cleaned["Airport.To"] == airport].copy()
    arr["scheduled_utc"] = pd.to_datetime(arr["Scheduled.Arrival.UTC"], errors="coerce")
    arr["actual_utc"]    = pd.to_datetime(arr["Arrival.UTC"], errors="coerce")
    arr = arr.dropna(subset=["scheduled_utc", "actual_utc"])
    arr["actual_delay_min"] = (arr["actual_utc"] - arr["scheduled_utc"]).dt.total_seconds() / 60
    arr["event_type"] = "arrival"
    rows.append(arr[["Flight_ID", "event_type", "scheduled_utc", "actual_utc", "actual_delay_min"]])

    return pd.concat(rows, ignore_index=True)


def evaluate(log_dir: str, schedule_file: str, cleaned_file: str, airport: str):
    print("=" * 70)
    print("Schedule Evaluation")
    print("=" * 70)

    schedule, cleaned = load_files(log_dir, schedule_file, cleaned_file)

    # 解析排班表时间
    schedule["planned_time"]   = pd.to_datetime(schedule["planned_time"])
    schedule["scheduled_time"] = pd.to_datetime(schedule["scheduled_time"])
    schedule["algo_delay_min"] = (
        schedule["scheduled_time"] - schedule["planned_time"]
    ).dt.total_seconds() / 60

    # 获取实际延误
    actual_df = compute_actual_delay(cleaned, airport)

    # 合并（flight_id + operation 对应）
    merged = schedule.merge(
        actual_df,
        left_on=["flight_id", "operation"],
        right_on=["Flight_ID", "event_type"],
        how="inner"
    )

    total_matched = len(merged)
    if total_matched == 0:
        print("\nWarning: No flights matched between schedule and cleaned data. "
              "Please check that flight_id and airport code are consistent.")
        return

    print(f"\nMatched flights: {total_matched} / {len(schedule)}")

    # ---- 原始计划延误（计划时间 vs 实际时间）----
    orig_delays = merged["actual_delay_min"]
    orig_positive = orig_delays[orig_delays > 0]

    # ---- 算法调度延误（调度时间 vs 实际时间）----
    merged["algo_vs_actual_min"] = (
        merged["scheduled_time"] - merged["actual_utc"]
    ).dt.total_seconds() / 60
    algo_delays = merged["algo_vs_actual_min"]
    algo_positive = algo_delays[algo_delays > 0]

    # ---- 算法相对计划的偏移 ----
    algo_offset = merged["algo_delay_min"]

    print("\n" + "-" * 70)
    print(f"{'Metric':<35} {'Original Plan':>15} {'Algorithm':>15}")
    print("-" * 70)

    def fmt(val):
        return f"{val:>15.2f}"

    print(f"{'Avg delay (all flights, min)':<35}"
          f"{fmt(orig_delays.mean())}{fmt(algo_delays.mean())}")
    print(f"{'Avg delay (delayed only, min)':<35}"
          f"{fmt(orig_positive.mean() if len(orig_positive) else 0)}"
          f"{fmt(algo_positive.mean() if len(algo_positive) else 0)}")
    print(f"{'Max delay (min)':<35}"
          f"{fmt(orig_delays.max())}{fmt(algo_delays.max())}")
    print(f"{'Delayed flights':<35}"
          f"{len(orig_positive):>15}{len(algo_positive):>15}")
    print(f"{'Delayed flight ratio':<35}"
          f"{len(orig_positive)/total_matched*100:>14.1f}%"
          f"{len(algo_positive)/total_matched*100:>14.1f}%")
    print(f"{'Total delay (min)':<35}"
          f"{fmt(orig_positive.sum())}{fmt(algo_positive.sum())}")
    print("-" * 70)

    # ---- 算法相对计划的偏移统计 ----
    print(f"\nAlgorithm schedule vs planned time:")
    print(f"  On-time flights:  {(algo_offset == 0).sum()}")
    print(f"  Delayed flights:  {(algo_offset > 0).sum()}, avg delay {algo_offset[algo_offset > 0].mean():.2f} minutes")
    print(f"  Early flights:    {(algo_offset < 0).sum()}")

    # ---- 总结 ----
    orig_total = orig_positive.sum()
    algo_total = algo_positive.sum()
    improvement = orig_total - algo_total
    pct = improvement / orig_total * 100 if orig_total > 0 else 0

    print("\n" + "=" * 70)
    if improvement > 0:
        print(f"Conclusion: Algorithm reduced total delay by {improvement:.2f} minutes (improved by {pct:.1f}%)")
    elif improvement < 0:
        print(f"Conclusion: Algorithm increased total delay by {abs(improvement):.2f} minutes (worsened by {abs(pct):.1f}%)")
    else:
        print("Conclusion: Algorithm schedule has the same total delay as the original plan")
    print("=" * 70)


if __name__ == "__main__":
    evaluate(LOG_DIR, SCHEDULE_FILE, CLEANED_FILE, AIRPORT_CODE)
