# backend/services/data_processor.py
# 步骤 1.2：多源数据接入与处理（论文 2.2）
# 统一：数据源配置、拉取、对齐到 10 分钟网格、缺失值填充、模型用列序

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 配置：API 与设备 ID（与 prediction_service / 论文一致）
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get(
    "SHM_API_BASE_URL",
    "http://139.159.136.213:4999/iem/shm",
)

# 数据源路径（多源接入）
SOURCE_PATHS = {
    "crack": "jmData",      # 测缝计
    "tilt": "jmBus",        # 测斜探头
    "level": "jmLevel",     # 水准/沉降
    "water_level": "jmWlg", # 水位
}

# 设备编号（与训练/预测一致）
CRACK_NUMBERS = ["623622", "623641", "623628"]
SETTLEMENT_NUMBERS = ["004521", "004548", "004591", "152947"]
TILT_NUMBERS = ["00476464", "00476465", "00476466", "00476467"]
WATER_LEVEL_NUMBER = "478967"

# 模型用列序（与 prediction_service.process_sensor_data_to_dataframe 一致）
COLUMNS_ORDER = (
    [f"settlement_{i+1}" for i in range(4)]
    + [f"crack_{i+1}" for i in range(3)]
    + [f"tilt_x_{i+1}" for i in range(4)]
    + [f"tilt_y_{i+1}" for i in range(4)]
    + ["water_level", "temperature"]
)

# 时间对齐：10 分钟间隔
ALIGN_INTERVAL_MINUTES = 10


# ---------------------------------------------------------------------------
# 多源拉取
# ---------------------------------------------------------------------------

def get_timestamp_range(
    hours_back: float = 24,
    end_time: Optional[datetime] = None,
) -> Tuple[int, int]:
    """返回 (timestamp1, timestamp2) 秒级时间戳。"""
    end = end_time or datetime.now()
    start = end - timedelta(hours=hours_back)
    return int(start.timestamp()), int(end.timestamp())


def fetch_source(
    source_key: str,
    timestamp1: int,
    timestamp2: int,
    base_url: str = API_BASE_URL,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """
    拉取单一数据源。
    source_key: crack | tilt | level | water_level
    返回原始记录列表。
    """
    path = SOURCE_PATHS.get(source_key)
    if not path:
        return []
    url = f"{base_url.rstrip('/')}/{path}"
    try:
        r = requests.get(
            url,
            params={"timestamp1": timestamp1, "timestamp2": timestamp2},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def fetch_all_sources(
    timestamp1: int,
    timestamp2: int,
    base_url: str = API_BASE_URL,
    timeout: int = 15,
) -> Dict[str, List[Dict[str, Any]]]:
    """拉取全部四类数据源，返回 { source_key: [raw_records] }。"""
    return {
        key: fetch_source(key, timestamp1, timestamp2, base_url, timeout)
        for key in SOURCE_PATHS
    }


# ---------------------------------------------------------------------------
# 时间网格（10 分钟对齐）
# ---------------------------------------------------------------------------

def round_time_to_interval(dt: datetime, interval_minutes: int = ALIGN_INTERVAL_MINUTES) -> datetime:
    """将时间对齐到 interval_minutes 的整格（四舍五入）。"""
    minutes = dt.minute
    rounded = ((minutes // interval_minutes) + (1 if minutes % interval_minutes >= interval_minutes // 2 else 0)) * interval_minutes
    if rounded >= 60:
        return (dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return dt.replace(minute=rounded, second=0, microsecond=0)


def build_time_index(
    start: datetime,
    end: datetime,
    interval_minutes: int = ALIGN_INTERVAL_MINUTES,
) -> pd.DatetimeIndex:
    """生成 [start, end] 内按 interval_minutes 对齐的时间索引。"""
    points: List[datetime] = []
    # 对齐起点
    start_min = start.minute
    r = ((start_min // interval_minutes) + (1 if start_min % interval_minutes >= interval_minutes // 2 else 0)) * interval_minutes
    if r >= 60:
        current = start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        current = start.replace(minute=r, second=0, microsecond=0)
    while current <= end:
        points.append(current)
        current += timedelta(minutes=interval_minutes)
    return pd.DatetimeIndex(points)


# ---------------------------------------------------------------------------
# 原始记录 → 对齐到时间网格的 DataFrame
# ---------------------------------------------------------------------------

def _parse_ts(record: Dict[str, Any]) -> Optional[datetime]:
    ts = record.get("timestamp")
    if ts is None:
        return None
    t = int(ts) if isinstance(ts, str) else ts
    return datetime.fromtimestamp(t)


def _align_and_put(
    data_dict: Dict[str, List],
    time_str: str,
    time_list: List[str],
    col: str,
    value: float,
) -> None:
    if time_str not in time_list:
        return
    idx = time_list.index(time_str)
    data_dict[col][idx] = value


def raw_to_fused_dataframe(
    raw: Dict[str, List[Dict[str, Any]]],
    start: datetime,
    end: datetime,
    interval_minutes: int = ALIGN_INTERVAL_MINUTES,
) -> pd.DataFrame:
    """
    将多源原始记录融合为一张按 10 分钟对齐的 DataFrame，列序与模型一致。
    """
    time_index = build_time_index(start, end, interval_minutes)
    time_list = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_index]

    data_dict: Dict[str, List] = {"time": time_list}
    for col in COLUMNS_ORDER:
        data_dict[col] = [None] * len(time_list)

    # 测缝计
    for item in raw.get("crack", []):
        num = str(item.get("number", ""))
        if num not in CRACK_NUMBERS:
            continue
        dt = _parse_ts(item)
        if not dt:
            continue
        dt = round_time_to_interval(dt, interval_minutes)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        idx_col = CRACK_NUMBERS.index(num) + 1
        val = item.get("data1")
        if val is not None:
            try:
                _align_and_put(data_dict, time_str, time_list, f"crack_{idx_col}", float(val))
            except (TypeError, ValueError):
                pass

    # 沉降/水准
    for item in raw.get("level", []):
        num = str(item.get("number", ""))
        if num not in SETTLEMENT_NUMBERS:
            continue
        dt = _parse_ts(item)
        if not dt:
            continue
        dt = round_time_to_interval(dt, interval_minutes)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        idx_col = SETTLEMENT_NUMBERS.index(num) + 1
        val = item.get("data1")
        if val is not None:
            try:
                _align_and_put(data_dict, time_str, time_list, f"settlement_{idx_col}", float(val))
            except (TypeError, ValueError):
                pass

    # 测斜（X/Y + 温度取第一个探头 data3）
    for item in raw.get("tilt", []):
        num = str(item.get("number", ""))
        if num not in TILT_NUMBERS:
            continue
        dt = _parse_ts(item)
        if not dt:
            continue
        dt = round_time_to_interval(dt, interval_minutes)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        idx_col = TILT_NUMBERS.index(num) + 1
        for col_suffix, key in [("x", "data1"), ("y", "data2")]:
            val = item.get(key)
            if val is not None:
                try:
                    _align_and_put(data_dict, time_str, time_list, f"tilt_{col_suffix}_{idx_col}", float(val))
                except (TypeError, ValueError):
                    pass
        if idx_col == 1:
            val_temp = item.get("data3")
            if val_temp is not None:
                try:
                    _align_and_put(data_dict, time_str, time_list, "temperature", float(val_temp))
                except (TypeError, ValueError):
                    pass

    # 水位（设备号可能带前导 0，如 478967 / 0478967）
    for item in raw.get("water_level", []):
        num_raw = str(item.get("number", ""))
        num_norm = num_raw.lstrip("0") or "0"
        target_norm = WATER_LEVEL_NUMBER.lstrip("0") or "0"
        if num_norm != target_norm and num_raw != WATER_LEVEL_NUMBER:
            continue
        dt = _parse_ts(item)
        if not dt:
            continue
        dt = round_time_to_interval(dt, interval_minutes)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        val = item.get("data1")
        if val is not None:
            try:
                _align_and_put(data_dict, time_str, time_list, "water_level", float(val))
            except (TypeError, ValueError):
                pass

    df = pd.DataFrame(data_dict)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df = df[COLUMNS_ORDER]
    return df


# ---------------------------------------------------------------------------
# 缺失值填充（论文 2.2：保证模型输入完整）
# ---------------------------------------------------------------------------

def fill_missing(
    df: pd.DataFrame,
    method: str = "forward-backward",
    max_gap_minutes: Optional[int] = 60,
    fallback_value: float = 0.0,
) -> pd.DataFrame:
    """
    统一缺失值填充。
    method: 'forward-backward' | 'linear' | 'forward' | 'backward'
    max_gap_minutes: 超过此间隔不插值，保持/填 fallback；None 表示不限制。
    fallback_value: 最后仍缺失时填充的值。
    """
    out = df.copy()
    if out.index.name != "time" and "time" in out.columns:
        out = out.set_index("time")
    out = out.sort_index()

    for col in out.columns:
        if out[col].isna().all():
            out[col] = fallback_value
            continue
        if method == "forward-backward":
            out[col] = out[col].ffill().bfill()
        elif method == "forward":
            out[col] = out[col].ffill()
        elif method == "backward":
            out[col] = out[col].bfill()
        elif method == "linear":
            out[col] = out[col].interpolate(method="linear", limit_direction="both")
        else:
            out[col] = out[col].ffill().bfill()
        out[col] = out[col].fillna(fallback_value)
    return out


# ---------------------------------------------------------------------------
# 一站式管道：拉取 → 融合 → 填充
# ---------------------------------------------------------------------------

@dataclass
class DataProcessorConfig:
    """可注入的配置，便于测试或切换环境。"""
    base_url: str = field(default_factory=lambda: API_BASE_URL)
    timeout: int = 15
    interval_minutes: int = ALIGN_INTERVAL_MINUTES
    fill_method: str = "forward-backward"
    max_gap_minutes: Optional[int] = 60
    fallback_value: float = 0.0


def process(
    hours_back: float = 24,
    end_time: Optional[datetime] = None,
    config: Optional[DataProcessorConfig] = None,
    raw_provider: Optional[Callable[[int, int], Dict[str, List[Dict[str, Any]]]]] = None,
) -> pd.DataFrame:
    """
    多源数据接入与处理一站式接口（论文 2.2）。
    - 若提供 raw_provider(timestamp1, timestamp2)，则用其返回的 raw 替代 API 拉取（便于测试/离线）。
    - 否则按 config.base_url 拉取 API，再融合、填充后返回。
    """
    cfg = config or DataProcessorConfig()
    ts1, ts2 = get_timestamp_range(hours_back=hours_back, end_time=end_time)
    end_dt = end_time or datetime.now()
    start_dt = end_dt - timedelta(hours=hours_back)

    if raw_provider is not None:
        raw = raw_provider(ts1, ts2)
    else:
        raw = fetch_all_sources(ts1, ts2, base_url=cfg.base_url, timeout=cfg.timeout)

    df = raw_to_fused_dataframe(raw, start_dt, end_dt, interval_minutes=cfg.interval_minutes)
    df = fill_missing(
        df,
        method=cfg.fill_method,
        max_gap_minutes=cfg.max_gap_minutes,
        fallback_value=cfg.fallback_value,
    )
    return df


# ---------------------------------------------------------------------------
# 样本数据（CSV）加载
# ---------------------------------------------------------------------------

def load_sample_data(
    name: str = "yuhuangge_sample",
    sample_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """从 backend/sample_data/ 加载示例 CSV；若不存在返回 None。"""
    if sample_dir is None:
        sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sample_data",
        )
    path = os.path.join(sample_dir, f"{name}.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    return df


# ---------------------------------------------------------------------------
# 归一化记录（可选：供 API 返回统一结构）
# ---------------------------------------------------------------------------

def normalize_records(
    records: List[Dict[str, Any]],
    sensor_type: str,
) -> List[Dict[str, Any]]:
    """
    将原始 API 记录规范为统一结构，便于前端或告警使用。
    sensor_type: crack | tilt | level | water_level
    返回列表，每项含 time, timestamp, sensor_id, value(s)。
    """
    out = []
    for r in records:
        ts = _parse_ts(r)
        if not ts:
            continue
        base = {"time": ts.strftime("%Y-%m-%d %H:%M:%S"), "timestamp": int(ts.timestamp())}
        num = str(r.get("number", ""))
        if sensor_type == "crack":
            if num not in CRACK_NUMBERS:
                continue
            base["sensor_id"] = num
            base["value"] = r.get("data1")
        elif sensor_type == "level":
            if num not in SETTLEMENT_NUMBERS:
                continue
            base["sensor_id"] = num
            base["value"] = r.get("data1")
        elif sensor_type == "tilt":
            if num not in TILT_NUMBERS:
                continue
            base["sensor_id"] = num
            base["value_x"] = r.get("data1")
            base["value_y"] = r.get("data2")
            base["value_temperature"] = r.get("data3")
        elif sensor_type == "water_level":
            base["sensor_id"] = num
            base["value"] = r.get("data1")
        else:
            base["sensor_id"] = num
            base["value"] = r.get("data1")
        out.append(base)
    return out
