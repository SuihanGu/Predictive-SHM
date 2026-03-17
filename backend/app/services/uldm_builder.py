# -*- coding: utf-8 -*-
"""
从原始传感器记录构建 ULDM（通用逻辑数据模型）。
与 DataProcessor 共用列序与重采样逻辑，输出 time_index、targets、covariates、full_matrix。
"""
from __future__ import annotations

import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd

from app.schemas.uldm import ULDM

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# 默认列序，与 data_processor 一致
_DEFAULT_COLUMNS_ORDER = (
    [f"settlement_{i+1}" for i in range(4)]
    + [f"crack_{i+1}" for i in range(3)]
    + [f"tilt_x_{i+1}" for i in range(4)]
    + [f"tilt_y_{i+1}" for i in range(4)]
    + ["water_level", "temperature"]
)


def _get_config_columns() -> tuple:
    """从 model_config 获取 columns_order、response_columns、env_columns"""
    try:
        from models.config import ModelConfig
        cfg = ModelConfig.from_json(os.path.join(_BACKEND_ROOT, "models", "model_config.json"))
        order = tuple(cfg.columns_order()) if cfg.columns_order() else _DEFAULT_COLUMNS_ORDER
        resp = list(cfg.response_columns()) if hasattr(cfg, "response_columns") else [f"crack_{i+1}" for i in range(3)]
        env = list(cfg.env_columns()) if hasattr(cfg, "env_columns") else ["water_level", "temperature"]
        return order, resp, env
    except Exception:
        order = _DEFAULT_COLUMNS_ORDER
        resp = [f"crack_{i+1}" for i in range(3)]
        env = ["water_level", "temperature"]
        return order, resp, env


def build_uldm(
    raw_records: List[Dict],
    resample_min: int = 10,
) -> ULDM:
    """
    将原始传感器记录转为 ULDM。
    与 DataProcessor 一致：按 timestamp/time 排序、重采样、列序补全。
    """
    order, response_columns, env_columns = _get_config_columns()

    df = pd.DataFrame(raw_records)
    if "timestamp" not in df.columns and "time" not in df.columns:
        raise ValueError("数据必须包含 timestamp 或 time 字段")

    ts_col = "timestamp" if "timestamp" in df.columns else "time"
    if df[ts_col].dtype in (np.int64, np.float64, int, float):
        df[ts_col] = pd.to_datetime(df[ts_col], unit="s", utc=True).dt.tz_localize(None)
    else:
        df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).set_index(ts_col)
    df = df.resample(f"{resample_min}min").mean()
    df = df.interpolate(method="linear").ffill().bfill()

    for c in order:
        if c not in df.columns:
            df[c] = 0.0
    cols = [c for c in order if c in df.columns]
    df = df[cols]

    time_index = df.index.values
    full_matrix = df.values.astype(np.float32)

    target_cols = [c for c in response_columns if c in df.columns]
    cov_cols = [c for c in env_columns if c in df.columns]
    targets = df[target_cols].values.astype(np.float32) if target_cols else np.zeros((len(df), 0), dtype=np.float32)
    covariates = df[cov_cols].values.astype(np.float32) if cov_cols else np.zeros((len(df), 0), dtype=np.float32)

    return ULDM(
        time_index=time_index,
        targets=targets,
        covariates=covariates,
        static_meta={"resample_min": resample_min},
        full_matrix=full_matrix,
    )
