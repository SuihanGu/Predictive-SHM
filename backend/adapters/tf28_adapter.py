# backend/adapters/tf28_adapter.py
# TensorFlow 2.8 模型适配器：加载 SavedModel / .h5/.keras，统一 predict 接口

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


class ModelAdapterBase(ABC):
    """与 model_adapter.ModelAdapter 一致的接口，避免未装 torch 时导入 model_adapter 报错。"""

    @abstractmethod
    def predict(self, x: Union[np.ndarray, tuple]) -> np.ndarray:
        pass


_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)


class TF28Adapter(ModelAdapterBase):
    """
    使用 TensorFlow 2.8 加载的模型适配器。
    支持：SavedModel 目录、.h5 / .keras 文件。
    输入/输出形状由您提供的模型决定，predict(x) 传入 numpy 数组即可。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_all_path: Optional[str] = None,
        scaler_response_path: Optional[str] = None,
        **kwargs,
    ):
        if not _TF_AVAILABLE:
            raise RuntimeError("请先安装 TensorFlow 2.8：pip install tensorflow>=2.8,<2.16")
        if model_path is None:
            # 默认尝试 backend/models 下常见 TF 格式
            for name in ["saved_model", "model.keras", "model.h5"]:
                p = os.path.join(_DEFAULT_MODEL_DIR, name)
                if os.path.exists(p):
                    model_path = p
                    break
            if model_path is None:
                model_path = os.path.join(_DEFAULT_MODEL_DIR, "saved_model")
        self._path = model_path
        self._model = self._load_model(model_path)
        self._scaler_all = None
        self._scaler_response = None
        if scaler_all_path and os.path.isfile(scaler_all_path):
            import pickle
            with open(scaler_all_path, "rb") as f:
                self._scaler_all = pickle.load(f)
        if scaler_response_path and os.path.isfile(scaler_response_path):
            import pickle
            with open(scaler_response_path, "rb") as f:
                self._scaler_response = pickle.load(f)

    def _load_model(self, path: str):
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "saved_model.pb")):
            return tf.keras.models.load_model(path)
        if path.endswith(".keras") or path.endswith(".h5"):
            return tf.keras.models.load_model(path)
        raise FileNotFoundError(f"未找到 TF 模型：{path}（支持 SavedModel 目录、.keras、.h5）")

    def predict(self, x: Union[np.ndarray, tuple]) -> np.ndarray:
        """
        x: 单数组 [batch, ...] 或 多输入 tuple，与您模型输入一致。
        返回: [batch, pred_steps, ...]，若配置了 scaler_response 会做逆变换。
        """
        if isinstance(x, (tuple, list)):
            inp = [np.asarray(t, dtype=np.float32) for t in x]
            out = self._model.predict(inp, verbose=0)
        else:
            inp = np.asarray(x, dtype=np.float32)
            out = self._model.predict(inp, verbose=0)
        out = np.asarray(out)
        if self._scaler_response is not None and out.ndim >= 2:
            batch = out.shape[0]
            flat = out.reshape(-1, out.shape[-1])
            out = self._scaler_response.inverse_transform(flat)
            out = out.reshape(batch, -1, out.shape[-1])
        return out


# 注册到可插拔列表（应用层 from adapters.model_adapter import MODEL_REGISTRY 即可看到 tf28）
def _register_tf28():
    try:
        from .model_adapter import MODEL_REGISTRY
        MODEL_REGISTRY["tf28"] = TF28Adapter
    except Exception:
        pass


if _TF_AVAILABLE:
    _register_tf28()
