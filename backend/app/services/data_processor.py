# Data processing service (paper §2.2)
# 多源传感器 → 统一时序张量，供 predict / data 路由使用
# 列序由 model_config.json 配置，支持用户自定义接入设备

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional
import os

# 默认列序（与原有 training_data 一致），可被 config 覆盖
_DEFAULT_COLUMNS_ORDER = (
    [f"settlement_{i+1}" for i in range(4)]
    + [f"crack_{i+1}" for i in range(3)]
    + [f"tilt_x_{i+1}" for i in range(4)]
    + [f"tilt_y_{i+1}" for i in range(4)]
    + ["water_level", "temperature"]
)


def _get_columns_order() -> tuple:
    """从 model_config 获取列序，失败时使用默认"""
    try:
        import sys
        _backend = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _backend not in sys.path:
            sys.path.insert(0, _backend)
        from models.config import ModelConfig
        cfg = ModelConfig.from_json(os.path.join(_backend, "models", "model_config.json"))
        return tuple(cfg.columns_order()) if cfg.columns_order() else _DEFAULT_COLUMNS_ORDER
    except Exception:
        return _DEFAULT_COLUMNS_ORDER


COLUMNS_ORDER = _get_columns_order()


class DataProcessor:
    """将原始传感器记录转为模型输入张量 [batch, time_steps, feature_dim]。"""

    def __init__(self):
        self.scaler = StandardScaler()

    def process(self, raw_records: List[Dict], resample_min: int = 10) -> np.ndarray:
        """多源传感器 → 统一时序张量 [1, time_steps, 17]，列序与 Transformer-CNN 一致"""
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

        for c in COLUMNS_ORDER:
            if c not in df.columns:
                df[c] = 0.0
        cols = [c for c in COLUMNS_ORDER if c in df.columns]
        df = df[cols]
        # 不在此处标准化，由 model adapter 使用 scaler_all 处理
        tensor = df.values.astype(np.float32)
        return tensor.reshape(1, tensor.shape[0], tensor.shape[1])
