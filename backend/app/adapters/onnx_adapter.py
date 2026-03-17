# -*- coding: utf-8 -*-
"""
ONNX 模型适配器：加载 .onnx 模型进行推理，支持用户自定义模型。
用户将 model.onnx 与 meta.json 放入 backend/models/user_models/{model_id}/ 并在 model_registry.json 中注册即可。
"""
from __future__ import annotations

import json
import numpy as np
import os
from typing import Optional

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    ort = None


def get_onnx_adapter_class():
    """若 onnxruntime 已安装则返回 ONNXAdapter 类，否则返回 None"""
    if not _ONNX_AVAILABLE:
        return None

    from app.adapters.base import ModelAdapter
    from app.schemas.uldm import ULDM

    class ONNXAdapter(ModelAdapter):
        """ONNX 模型适配器，输入 [batch, time_steps, features] → 输出 [batch, pred_steps, target_dim]"""

        def __init__(
            self,
            model_path: str,
            meta_path: Optional[str] = None,
            target_sensor: str = "",
            output_dim: int = 1,
            pred_steps: int = 6,
        ):
            self.model_path = model_path
            self.target_sensor = target_sensor
            self.output_dim = output_dim
            self.pred_steps = pred_steps
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._meta = {}
            if meta_path and os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)

        def from_uldm(self, uldm: ULDM) -> np.ndarray:
            """使用 full_matrix 或 targets|covariates 构造 [1, T, features]"""
            if uldm.full_matrix is not None:
                x = np.asarray(uldm.full_matrix, dtype=np.float32)
            else:
                x = np.hstack([uldm.targets, uldm.covariates]).astype(np.float32)
            return x[np.newaxis, :, :]

        def predict(self, x: np.ndarray) -> np.ndarray:
            x = np.atleast_3d(x).astype(np.float32)
            # ONNX 输入格式依模型而定，此处假设 [batch, seq, features]
            out = self._session.run(None, {self._input_name: x})[0]
            out = np.asarray(out, dtype=np.float32)
            # 确保输出形状 [batch, pred_steps, output_dim]
            if out.ndim == 2:
                out = out.reshape(1, -1, self.output_dim)
            if out.shape[2] != self.output_dim:
                out = out.reshape(out.shape[0], -1, self.output_dim)
            return out

        def get_capabilities(self) -> dict:
            return {
                "min_time_steps": 1,
                "max_pred_steps": self.pred_steps,
                "supports_uncertainty": False,
                "target_sensor_types": [self.target_sensor] if self.target_sensor else [],
            }

        def get_meta(self) -> dict:
            return {
                "target_sensor": self.target_sensor,
                "output_dim": self.output_dim,
                "pred_steps": self.pred_steps,
                **self._meta,
            }

        def supports_sensor(self, key: str) -> bool:
            return key == self.target_sensor or not self.target_sensor

    return ONNXAdapter
