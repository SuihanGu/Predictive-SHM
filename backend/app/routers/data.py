from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any, Optional
import os
import time
from collections import deque
import pandas as pd
from app.services.data_processor import DataProcessor
from app.services.config_loader import get_sensors
from app.services.data_format import normalize_record

router = APIRouter()
processor = DataProcessor()

# 实时传感器数据缓冲（最多保留 2000 条，供真实传感器接入）
_REALTIME_BUFFER: deque = deque(maxlen=2000)

# 真实样例数据路径（backend/sample_data/training_data.csv）
_SAMPLE_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "sample_data", "training_data.csv"
)


def _gen(i: int, sensors: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    """
    生成“虚拟样例数据”的单行数值（归一化前）。
    实际工程中传感器/通道不固定：这里按 monitor_config.json 的 sensors 动态生成字段，
    避免写死 crack_1/tilt_x_1/... 等键。
    """
    sensors = sensors or get_sensors()

    base = 0.1 + 0.02 * (i % 10)
    drift = 0.005 * (i % 7)
    env_base = 20 + 2 * (i % 5)

    vals: Dict[str, float] = {}
    for s in sensors:
        key = s.get("key") or ""
        channels = s.get("channels") or []
        data_key = s.get("data_key") or key

        def _v(j: int) -> float:
            # 让不同通道有轻微差异、且随时间缓慢变化
            if "temp" in key:
                return float(round(env_base + 0.2 * j, 2))
            if "water" in key:
                return float(round(50 + 5 * (i % 8) + j * 0.5, 2))
            if "settle" in key:
                return float(round(0.5 + base + drift + j * 0.02, 3))
            if "tilt" in key:
                return float(round(0.01 + base * 0.2 + drift * 0.1 + j * 0.002, 4))
            # default (crack/strain/others)
            return float(round(base + drift + j * 0.01, 4))

        if channels:
            for j, ch in enumerate(channels):
                vals[ch] = _v(j)
            # 兼容：data_key 取首通道
            if data_key and data_key not in vals and channels:
                vals[data_key] = vals.get(channels[0], _v(0))
        else:
            vals[data_key] = _v(0)

    return vals


@router.post("/data/process")
async def process_data(records: List[Dict]):
    try:
        tensor = processor.process(records)
        return {"tensor_shape": list(tensor.shape), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/data/ingest")
async def ingest_sensor_data(body: Any = Body(...)):
    """
    实时传感器数据接入。
    请求体：单条对象 `{...}` 或 数组 `[{...},{...}]`
    支持两种格式：
    - 标准化：{ "ts": 1730350800, "data": { "k": v, ... } }（data 为任意传感器键值对）
    - 扁平（兼容）：{ "timestamp": 1730350800, "k": v, ... }
    数据写入内存缓冲，GET /api/data/sample 优先返回缓冲数据。
    """
    records = body if isinstance(body, list) else [body]
    if not isinstance(records, list):
        raise HTTPException(status_code=400, detail="需传入对象或数组")
    cnt = 0
    for r in records:
        try:
            row = normalize_record(r)
            _REALTIME_BUFFER.append(row)
            cnt += 1
        except Exception:
            continue
    return {"status": "ok", "ingested": cnt, "buffer_size": len(_REALTIME_BUFFER)}


def _load_real_sample() -> Optional[List[Dict[str, Any]]]:
    """加载 backend/sample_data/training_data.csv，支持任意列名（与 config.data_key 对应）"""
    if not os.path.isfile(_SAMPLE_CSV):
        return None
    try:
        df = pd.read_csv(_SAMPLE_CSV)
        # 时间列：优先 time/timestamp，否则首列
        time_col = None
        for c in ("time", "timestamp"):
            if c in df.columns:
                time_col = c
                break
        if time_col is None and len(df.columns) > 0:
            time_col = df.columns[0]
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).tail(500)
        records = []
        for _, r in df.iterrows():
            ts_val = r[time_col]
            ts = int(ts_val.timestamp()) if hasattr(ts_val, "timestamp") else int(pd.Timestamp(ts_val).timestamp())
            row: Dict[str, Any] = {"timestamp": ts}
            for c in df.columns:
                if c == time_col:
                    continue
                try:
                    val = r[c]
                    if pd.isna(val):
                        row[c] = 0.0
                    else:
                        row[c] = float(val)
                except (TypeError, ValueError):
                    pass
            # 为 monitor_config 的 data_key 提供兼容：crack/tilt_x 等指向首通道
            if "crack_1" in row and "crack" not in row:
                row["crack"] = row.get("crack_1", 0)
            if "tilt_x_1" in row and "tilt_x" not in row:
                row["tilt_x"] = row.get("tilt_x_1", 0)
            if "tilt_y_1" in row and "tilt_y" not in row:
                row["tilt_y"] = row.get("tilt_y_1", 0)
            if "settlement_1" in row and "settlement" not in row:
                row["settlement"] = row.get("settlement_1", 0)
            records.append(row)
        return records
    except Exception:
        return None


@router.get("/data/sample")
async def get_sample_data():
    """
    数据优先级：1. 实时接入缓冲  2. 样例 CSV  3. 虚拟数据
    """
    if len(_REALTIME_BUFFER) >= 60:
        return list(_REALTIME_BUFFER)
    real = _load_real_sample()
    if real:
        return real
    import time
    now = int(time.time())
    sensors = get_sensors()
    virtual = []
    for i in range(144, 0, -1):
        t = now - i * 600
        vals = _gen(i, sensors)
        row: Dict[str, Any] = {"timestamp": t, **vals}
        # 兼容字段：为未指定 data_key 的旧前端/脚本提供一个 data1
        if "data1" not in row:
            # 取任意一个数值字段作为 data1，尽量选择第一个传感器的第一个通道/数据键
            if sensors:
                s0 = sensors[0]
                ch0 = (s0.get("channels") or [None])[0]
                dk0 = s0.get("data_key") or s0.get("key")
                pick = ch0 or dk0
                if pick and pick in row:
                    row["data1"] = row[pick]
        # 兼容：若某传感器配置了 data_key，确保该键存在（等价于首通道）
        for s in sensors:
            dk = s.get("data_key") or s.get("key")
            chs = s.get("channels") or []
            if dk and dk not in row and chs:
                row[dk] = row.get(chs[0], 0.0)
        virtual.append(row)
    return virtual

__all__ = ["router"]