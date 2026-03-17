"""
模型适配器：Transformer-CNN、ONNX 等预测模型的封装。
支持多源传感器，通过 model_registry.json 与 model_config.json 配置。
支持 ULDM 输入与 StandardPrediction 输出（可插拔预测模块）。
"""
import pickle
import json
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.adapters.base import ModelAdapter, MockAdapter
from app.schemas.uldm import ULDM, StandardPrediction

MODEL_REGISTRY: Dict = {}

# 确保 backend 在 path 中
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    import torch
    from models.transformer_cnn import TransformerCnn
    from models.config import ModelConfig, DEFAULT_CONFIG_PATH
    from models.dataset import get_column_indices
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    TransformerCnn = None
    ModelConfig = None
    DEFAULT_CONFIG_PATH = ""
    get_column_indices = None


def _backend_models_path(*parts: str) -> str:
    return os.path.join(_BACKEND_ROOT, "models", *parts)


def get_onnx_adapter_class():
    """若 onnxruntime 已安装则返回 ONNXAdapter 类，否则返回 None"""
    try:
        from app.adapters.onnx_adapter import get_onnx_adapter_class as _get
        return _get()
    except Exception:
        return None


def _load_adapter_config() -> Optional[tuple]:
    """从 model_config.json 加载配置，返回 (m, n, lag, response_dim, env_dim, trans_dim, ri, ei, ti)"""
    config_path = os.environ.get("MODEL_CONFIG", _backend_models_path("model_config.json"))
    m, n, lag = 30, 6, 80
    response_dim, env_dim, trans_dim = 3, 2, 15
    ri, ei, ti = [4, 5, 6], [15, 16], list(range(15))

    if ModelConfig is not None and get_column_indices is not None and os.path.isfile(config_path):
        config = ModelConfig.from_json(config_path)
        m, n, lag = config.m, config.n, config.lag
        response_dim = config.response_dim
        env_dim = config.env_dim
        trans_dim = config.trans_dim
        ri, ei, ti = get_column_indices(config)
    else:
        meta_path = _backend_models_path("model_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            m = meta.get("m", m)
            n = meta.get("n", n)
            lag = meta.get("lag", lag)
            response_dim = meta.get("response_dim", response_dim)
            env_dim = meta.get("env_dim", env_dim)
            trans_dim = meta.get("trans_dim", trans_dim)

    return (m, n, lag, response_dim, env_dim, trans_dim, ri, ei, ti)


if _TORCH_AVAILABLE and TransformerCnn is not None:

    class TransformerCNNAdapter(ModelAdapter):
        """加载 Transformer-CNN，支持多源传感器配置"""

        def __init__(
            self,
            model_path: Optional[str] = None,
            scaler_path: Optional[str] = None,
            response_scaler_path: Optional[str] = None,
        ):
            model_path = model_path or _backend_models_path("best_crack_model.pth")
            scaler_path = scaler_path or _backend_models_path("scaler_all.pkl")
            response_scaler_path = response_scaler_path or _backend_models_path("scaler_response.pkl")
            self.device = torch.device("cpu")
            cfg = _load_adapter_config()
            if cfg:
                self.M, self.N, self.LAG, self.RESPONSE_DIM, self.ENV_DIM, self.TRANS_DIM, self.ri, self.ei, self.ti = cfg
            else:
                self.M, self.N, self.LAG = 30, 6, 80
                self.RESPONSE_DIM, self.ENV_DIM, self.TRANS_DIM = 3, 2, 15
                self.ri, self.ei, self.ti = [4, 5, 6], [15, 16], list(range(15))

            self.model = self._load_model(model_path)
            self.model.eval()
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                with open(response_scaler_path, "rb") as f:
                    self.response_scaler = pickle.load(f)
            except FileNotFoundError:
                from sklearn.preprocessing import MinMaxScaler
                n_features = max(self.TRANS_DIM + self.ENV_DIM, 17)
                self.scaler = MinMaxScaler().fit(np.array([[0] * n_features, [1] * n_features]))
                self.response_scaler = MinMaxScaler().fit(np.array([[0] * self.RESPONSE_DIM, [1] * self.RESPONSE_DIM]))

        def _load_model(self, path: str):
            model = TransformerCnn(
                response_dim=self.RESPONSE_DIM,
                env_dim=self.ENV_DIM,
                trans_dim=self.TRANS_DIM,
                num_heads=3,
                ff_hidden_dim=128,
                conv_hidden_dim=96,
                kernel_size=3,
                dropout=0.25,
                n_steps=self.N,
                lag=self.LAG,
                m=self.M,
            )
            state = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(state)
            return model.to(self.device)

        def from_uldm(self, uldm: ULDM) -> np.ndarray:
            """从 ULDM 的 full_matrix 构造 [1, T, n_features]，满足 lag+m 长度供 predict 使用"""
            if uldm.full_matrix is None:
                raise ValueError("TransformerCNNAdapter 需要 ULDM.full_matrix")
            data = np.asarray(uldm.full_matrix, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            T, nf = data.shape[0], data.shape[1]
            m, lag = self.M, self.LAG
            need = lag + m
            if T < need:
                pad = np.zeros((need - T, nf), dtype=np.float32)
                data = np.vstack([pad, data])
            return data[np.newaxis, :, :].astype(np.float32)

        def predict(self, x: np.ndarray) -> np.ndarray:
            data = np.squeeze(x, axis=0)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            data = self.scaler.transform(data)
            T = data.shape[0]
            m, lag = self.M, self.LAG
            if T < lag + m:
                pad = np.zeros((lag + m - T, data.shape[1]), dtype=np.float32)
                data = np.vstack([pad, data])
            x_response = data[-m:, self.ri].astype(np.float32)
            x_env = data[-lag - m : -m, self.ei].astype(np.float32)
            x_cat = data[-m:, self.ti].astype(np.float32)
            x_r = torch.from_numpy(x_response[np.newaxis]).to(self.device)
            x_e = torch.from_numpy(x_env[np.newaxis]).to(self.device)
            x_c = torch.from_numpy(x_cat[np.newaxis]).to(self.device)
            with torch.no_grad():
                out = self.model(x_r, x_e, x_c).cpu().numpy()
            out_flat = out.reshape(-1, self.RESPONSE_DIM)
            return self.response_scaler.inverse_transform(out_flat).reshape(out.shape)

        def to_standard_output(
            self,
            raw_output: np.ndarray,
            base_time: datetime,
            step_minutes: int,
            sensor_ids: List[str],
            **kwargs: Any,
        ) -> StandardPrediction:
            """将 [batch, n_steps, response_dim] 转为带时间戳的 StandardPrediction"""
            ids = sensor_ids if sensor_ids else [f"crack_{i+1}" for i in range(self.RESPONSE_DIM)]
            return super().to_standard_output(
                raw_output, base_time, step_minutes, ids, **kwargs
            )

        def get_capabilities(self) -> dict:
            return {
                "min_time_steps": self.LAG + self.M,
                "max_pred_steps": self.N,
                "supports_uncertainty": False,
                "target_sensor_types": ["crack"],
                "time_interval_minutes": 10,
            }

        def get_meta(self) -> dict:
            return {
                "response_dim": self.RESPONSE_DIM,
                "n_steps": self.N,
                "m": self.M,
                "lag": self.LAG,
                **super().get_meta(),
            }

else:
    TransformerCNNAdapter = None
