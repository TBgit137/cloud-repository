import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import io, json, asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from src.data.validator import validate
from src.data.preprocessor import FlightDataPreprocessor
from src.algorithm.genetic_algorithm import GeneticAlgorithm

app = FastAPI(title="Airline Scheduler API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080",
                   "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
_executor = ThreadPoolExecutor(max_workers=2)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_flights(algorithm_input: dict) -> list:
    from datetime import datetime
    return [
        {
            'flight_id':    e['flight_id'],
            'planned_time': datetime.fromisoformat(e['scheduled_time']),
            'operation':    e['event_type'],
        }
        for e in algorithm_input['events']
    ]


def _run_ga(algorithm_input: dict, n_runways: int, safety_interval: int) -> dict:
    from src.algorithm.constraints import RunwayConstraints
    alg = GeneticAlgorithm(n_runways=n_runways)
    alg.constraints = RunwayConstraints(min_interval_minutes=safety_interval)
    return alg.optimize(_build_flights(algorithm_input))


def _serialize_schedule(schedule: list) -> list:
    rows = []
    for item in schedule:
        rows.append({
            'flight_id':      item.get('flight_id'),
            'operation':      item.get('operation', ''),
            'planned_time':   item['planned_time'].isoformat(),
            'scheduled_time': item['scheduled_time'].isoformat(),
            'delay_minutes':  round(
                (item['scheduled_time'] - item['planned_time']).total_seconds() / 60, 2
            ),
            'runway': item.get('runway'),
        })
    return rows


def _get_removed_flights(filtered_df: pd.DataFrame, clean_df: pd.DataFrame) -> list:
    """
    返回用户指定时段内、因异常被清洗掉的航班。
    filtered_df: 经过机场+时段筛选后的数据（清洗前）
    clean_df:    预处理完成后的干净数据
    """
    removed = []
    try:
        key_cols = ['Flight.No', 'Scheduled.Departure', 'Airport.From', 'Airport.To']
        keys = [c for c in key_cols if c in filtered_df.columns and c in clean_df.columns]
        if not keys:
            return []
        filtered_keys = set(filtered_df[keys].astype(str).apply(tuple, axis=1))
        clean_keys    = set(clean_df[keys].astype(str).apply(tuple, axis=1))
        removed_df = filtered_df[
            filtered_df[keys].astype(str).apply(tuple, axis=1).isin(filtered_keys - clean_keys)
        ]
        for _, row in removed_df.iterrows():
            removed.append({
                'flight_no':           str(row.get('Flight.No', '')),
                'airport_from':        str(row.get('Airport.From', '')),
                'airport_to':          str(row.get('Airport.To', '')),
                'scheduled_departure': str(row.get('Scheduled.Departure', '')),
                'scheduled_arrival':   str(row.get('Scheduled.Arrival', '')),
            })
    except Exception as e:
        print(f"[Warning] get removed flights failed: {e}")
    return removed


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    airport: str = Form("SBGR"),
    n_runways: int = Form(5),
    safety_interval: int = Form(3),
    start_date: str = Form(""),
    end_date: str = Form(""),
):
    content = await file.read()

    async def event_stream():
        loop = asyncio.get_event_loop()

        yield sse("status", {"step": "read", "message": "正在读取文件..."})
        try:
            original_df = pd.read_csv(io.BytesIO(content), encoding='latin1')
        except Exception as e:
            yield sse("error", {"message": f"文件读取失败: {e}"})
            return

        yield sse("status", {"step": "validate", "message": "正在验证数据集格式..."})
        passed, err = validate(original_df, airport)
        if not passed:
            yield sse("error", {"message": err})
            return

        yield sse("status", {"step": "preprocess", "message": "正在进行数据预处理（清洗、时区转换等）..."})
        try:
            preprocessor = FlightDataPreprocessor(log_dir=LOG_DIR)
            clean_df, algorithm_input, filtered_df = await loop.run_in_executor(
                _executor,
                partial(
                    preprocessor.process,
                    original_df.copy(), airport, n_runways, safety_interval,
                    start_date or None, end_date or None,
                )
            )
        except Exception as e:
            yield sse("error", {"message": f"预处理失败: {e}"})
            return

        if algorithm_input['total_events'] == 0:
            yield sse("error", {"message": "预处理后无有效航班事件，请检查机场代码或日期范围"})
            return

        removed_in_range = len(filtered_df) - len(clean_df)
        yield sse("status", {
            "step": "preprocess_done",
            "message": f"预处理完成，共 {algorithm_input['total_events']} 个事件（起飞 {algorithm_input['departure_events']}，降落 {algorithm_input['arrival_events']}），时段内清洗掉 {removed_in_range} 条记录",
        })

        yield sse("status", {"step": "optimize", "message": "正在运行遗传算法优化排班..."})
        try:
            result = await loop.run_in_executor(
                _executor,
                partial(_run_ga, algorithm_input, n_runways, safety_interval)
            )
        except Exception as e:
            yield sse("error", {"message": f"算法运行失败: {e}"})
            return

        yield sse("status", {"step": "finalize", "message": "正在整理排班结果..."})
        schedule_rows   = _serialize_schedule(result['schedule'])
        removed_flights = _get_removed_flights(filtered_df, clean_df)

        try:
            from datetime import datetime
            results_dir = os.path.join(LOG_DIR, "results")
            os.makedirs(results_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pd.DataFrame(schedule_rows).to_csv(
                os.path.join(results_dir, f"schedule_result_{ts}.csv"), index=False
            )
            with open(os.path.join(results_dir, f"schedule_summary_{ts}.json"), 'w', encoding='utf-8') as f:
                json.dump({
                    'algorithm': result['algorithm'],
                    'penalty': result['penalty'],
                    'total_scheduled': len(schedule_rows),
                    'airport': airport,
                    'n_runways': n_runways,
                    'safety_interval': safety_interval,
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] save result failed: {e}")

        yield sse("done", {
            "summary": {
                "airport":          algorithm_input['airport'],
                "n_runways":        algorithm_input['n_runways'],
                "total_events":     algorithm_input['total_events'],
                "departure_events": algorithm_input['departure_events'],
                "arrival_events":   algorithm_input['arrival_events'],
                "penalty":          round(result['penalty'], 2),
                "original_rows":    len(original_df),
                "filtered_rows":    len(filtered_df),
                "cleaned_rows":     len(clean_df),
                "removed_rows":     removed_in_range,
            },
            "schedule":        schedule_rows,
            "removed_flights": removed_flights,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )