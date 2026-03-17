# -*- coding: utf-8 -*-
"""
通用逻辑数据模型（ULDM）与标准预测输出（StandardPrediction）。
见 docs/可插拔预测模型修改方案.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class ULDM:
    """通用逻辑数据模型：平台内部时序预测输入的标准结构"""

    time_index: np.ndarray  # (T,) 采样时间点，可为 datetime 或 Unix 时间戳
    targets: np.ndarray  # (T, n_targets) 待预测的传感器物理量
    covariates: np.ndarray  # (T, n_covariates) 动态协变量，可为 (T, 0)
    static_meta: dict = field(default_factory=dict)  # 结构/传感器静态属性
    # 可选：完整特征矩阵 (T, n_features)，供需要按列序切分的模型（如 Transformer-CNN）使用
    full_matrix: Optional[np.ndarray] = None


@dataclass
class StandardPrediction:
    """工程友好的未来读数序列：带时间戳与可选不确定性"""

    time_index: List[Any]  # 预测时间点，ISO 字符串或 Unix 时间戳
    readings: np.ndarray  # (pred_steps, n_sensors)
    sensor_ids: List[str]  # 与 readings 列对应
    lower: Optional[np.ndarray] = None  # (pred_steps, n_sensors) 可选置信下界
    upper: Optional[np.ndarray] = None  # (pred_steps, n_sensors) 可选置信上界

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典"""
        readings_list = np.asarray(self.readings).tolist()
        lower_list = np.asarray(self.lower).tolist() if self.lower is not None else None
        upper_list = np.asarray(self.upper).tolist() if self.upper is not None else None
        return {
            "time_index": list(self.time_index),
            "readings": readings_list,
            "sensor_ids": list(self.sensor_ids),
            "lower": lower_list,
            "upper": upper_list,
        }
