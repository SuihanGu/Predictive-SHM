# -*- coding: utf-8 -*-
"""
标准化数据格式与转换
支持 ingest 接入的两种格式：
1. 标准化格式：{ "ts": 1730350800, "data": { "k": v, ... } }
   - data 为任意传感器键值对，不限于 crack/tilt，与 model_config 的 channels 对应即可
2. 扁平格式（兼容）：{ "timestamp": 1730350800, "k": v, ... }
统一转换为内部格式：{ "timestamp": int, ...columns }，并按 model_config 补全缺失列。
"""
from __future__ import annotations

import time
from typing import Dict, Any, List, Optional
import pandas as pd
import os

# 默认列序（config 加载失败时使用）
_DEFAULT_COLUMNS = (
    [f"settlement_{i+1}" for i in range(4)]
    + [f"crack_{i+1}" for i in range(3)]
    + [f"tilt_x_{i+1}" for i in range(4)]
    + [f"tilt_y_{i+1}" for i in range(4)]
    + ["water_level", "temperature"]
)


def _get_columns_order() -> tuple:
    """从 model_config 获取列序"""
    try:
        import sys
        _backend = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _backend not in sys.path:
            sys.path.insert(0, _backend)
        from models.config import ModelConfig
        cfg = ModelConfig.from_json(os.path.join(_backend, "models", "model_config.json"))
        return tuple(cfg.columns_order()) if cfg.columns_order() else _DEFAULT_COLUMNS
    except Exception:
        return _DEFAULT_COLUMNS


COLUMNS_ORDER = _get_columns_order()


def _parse_timestamp(val: Any) -> int:
    """将任意时间值解析为 Unix 秒级时间戳"""
    if val is None:
        return int(time.time())
    if hasattr(val, "timestamp"):
        return int(val.timestamp())
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return int(val)
    return int(pd.Timestamp(val).timestamp())


def to_flat_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单条上报转为扁平格式 { timestamp, ...data }。
    支持：
    - 标准化格式：{ "ts": 1730350800, "data": { "k": v, ... } }
      data 可为任意传感器键（如 settlement_1、strain_1、custom_sensor 等）
    - 扁平格式：{ "timestamp": 1730350800, "k": v, ... }
    """
    flat: Dict[str, Any] = {}
    ts_val = r.get("ts") or r.get("timestamp") or r.get("time")
    flat["timestamp"] = _parse_timestamp(ts_val)

    if "data" in r and isinstance(r["data"], dict):
        # 标准化格式：合并 data 中的键值
        for k, v in r["data"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[k] = float(v)
            elif v is not None and not isinstance(v, bool):
                try:
                    flat[k] = float(v)
                except (TypeError, ValueError):
                    pass
    else:
        # 扁平格式：复制除时间键外的数值列
        for k, v in r.items():
            if k in ("ts", "timestamp", "time"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[k] = float(v)
            elif v is not None and not isinstance(v, bool):
                try:
                    flat[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return flat


# 首通道别名（兼容 monitor_config.data_key）
_FIRST_CHANNEL_ALIASES = {
    "crack_1": ["crack", "data1"],
    "tilt_x_1": ["tilt_x"],
    "tilt_y_1": ["tilt_y", "data2"],
    "settlement_1": ["settlement"],
}


def _get_first_channel(col: str) -> Optional[str]:
    """crack_2 -> crack_1, water_level -> None"""
    if "_" in col:
        base, num = col.rsplit("_", 1)
        if num.isdigit() and base in ("crack", "tilt_x", "tilt_y", "settlement"):
            return f"{base}_1"
    return None


def normalize_record(r: Dict[str, Any], columns_order: Optional[tuple] = None) -> Dict[str, Any]:
    """
    将单条上报转为内部统一格式，按配置补全缺失列。
    返回：{ timestamp, ...columns }，含简化别名（crack, tilt_x, tilt_y, settlement）供 monitor 使用。
    """
    flat = to_flat_record(r)
    cols = columns_order or COLUMNS_ORDER

    row: Dict[str, Any] = {"timestamp": flat["timestamp"]}
    for c in cols:
        if c in flat:
            row[c] = float(flat[c])
        else:
            first_ch = _get_first_channel(c)
            val = flat.get(first_ch) if first_ch else None
            if val is None and first_ch and first_ch in _FIRST_CHANNEL_ALIASES:
                for alias in _FIRST_CHANNEL_ALIASES[first_ch]:
                    if alias in flat:
                        val = flat[alias]
                        break
            if val is not None:
                row[c] = float(val)
            else:
                row[c] = 20.0 if c == "temperature" else 0.0

    # 添加简化别名供 monitor 使用
    if "crack_1" in row:
        row.setdefault("crack", row["crack_1"])
        row.setdefault("data1", row["crack_1"])
    if "tilt_x_1" in row:
        row.setdefault("tilt_x", row["tilt_x_1"])
    if "tilt_y_1" in row:
        row.setdefault("tilt_y", row["tilt_y_1"])
    if "settlement_1" in row:
        row.setdefault("settlement", row["settlement_1"])

    # 保留用户自定义数值列（供 data_key 映射）
    for k, v in flat.items():
        if k in row or k == "timestamp":
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            row[k] = float(v)
    return row


def normalize_batch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量转换，支持单条对象或数组"""
    if not records:
        return []
    if isinstance(records, dict):
        records = [records]
    return [normalize_record(r) for r in records]
