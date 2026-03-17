# -*- coding: utf-8 -*-
"""模型适配器基类与 Mock 占位（ULDM + StandardPrediction 接口）"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, List, Optional

import numpy as np

from app.schemas.uldm import ULDM, StandardPrediction


class ModelAdapter(ABC):
    """预测模型适配器抽象基类：统一 predict，可选 from_uldm / to_standard_output / get_capabilities"""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """输入 [batch_size, time_steps, feature_dim] 或模型特定格式 → 输出 [batch_size, pred_steps, target_dim]"""
        pass

    def from_uldm(self, uldm: ULDM) -> Any:
        """将 ULDM 转为模型特定物理格式。未实现时由调用方传 raw 张量或子类实现。"""
        raise NotImplementedError("该适配器未实现 from_uldm，请使用兼容的原始张量输入")

    def to_standard_output(
        self,
        raw_output: np.ndarray,
        base_time: datetime,
        step_minutes: int,
        sensor_ids: List[str],
        **kwargs: Any,
    ) -> StandardPrediction:
        """将模型原始输出映射为带时间戳的 StandardPrediction。默认假设 raw_output 形状 (batch, pred_steps, n_sensors)。"""
        out = np.asarray(raw_output)
        if out.ndim == 2:
            out = out[np.newaxis, ...]
        batch, pred_steps, n_sensors = out.shape[0], out.shape[1], out.shape[2]
        readings = out[0] if batch >= 1 else out.reshape(0, n_sensors)
        ids = sensor_ids if len(sensor_ids) >= n_sensors else [f"sensor_{i+1}" for i in range(n_sensors)]
        time_index = [
            (base_time + timedelta(minutes=step_minutes * (i + 1))).isoformat()
            for i in range(pred_steps)
        ]
        return StandardPrediction(
            time_index=time_index,
            readings=readings,
            sensor_ids=ids[:n_sensors],
            lower=None,
            upper=None,
        )

    def get_capabilities(self) -> dict:
        """声明模型能力与需求，子类可覆盖。"""
        return {}

    def get_meta(self) -> dict:
        """返回模型元信息，子类可覆盖"""
        return {}

    def supports_sensor(self, key: str) -> bool:
        """是否支持某传感器，子类可覆盖"""
        return True


class MockAdapter(ModelAdapter):
    """无真实模型时的占位：返回基于输入的简单外推"""

    def from_uldm(self, uldm: ULDM) -> np.ndarray:
        """使用 full_matrix 或 targets 构造 [1, T, features] 供 predict"""
        if uldm.full_matrix is not None:
            x = np.asarray(uldm.full_matrix, dtype=np.float32)
        else:
            x = np.hstack([uldm.targets, uldm.covariates]).astype(np.float32)
        return x.reshape(1, x.shape[0], x.shape[1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_3d(x)
        if x.shape[0] == 0:
            return np.zeros((1, 6, 3), dtype=np.float32)
        last = x[0, -1, :]
        if last.size >= 3:
            base = np.array(
                [last[4], last[5], last[6]] if last.size > 6 else [last[0], last[1], last[2]],
                dtype=np.float32,
            )
        else:
            base = np.array([0.12, 0.12, 0.12], dtype=np.float32)
        n_steps = 6
        pred = base + np.linspace(0, 0.02, n_steps)[:, np.newaxis]
        return pred[np.newaxis].astype(np.float32)

    def get_capabilities(self) -> dict:
        return {"min_time_steps": 1, "max_pred_steps": 6, "supports_uncertainty": False}
