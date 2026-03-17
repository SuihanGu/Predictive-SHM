# backend/adapters/model_adapter.py
# 步骤 1.1：可插拔预测模型适配器（PyTorch 实现；torch 可选，未装时仅无 transformer_cnn）
# 使用 TensorFlow 2.8 模型时见 tf28_adapter.py，无需安装 torch

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# 抽象适配器接口
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """可插拔模型适配器基类：统一 predict 接口。"""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        推理接口。
        x: [batch, time_steps, feature_dim] 或由具体实现解释（如三输入模型可为 tuple）
        return: [batch, pred_steps, target_dim]
        """
        pass


# ---------------------------------------------------------------------------
# Transformer-CNN 模型定义（与训练 / prediction_service 一致）- 仅当 torch 已安装时定义
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _TransformerEncoderLayer(nn.Module):
        def __init__(self, input_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads
            )
            self.self_ff = nn.Sequential(
                nn.Linear(input_dim, ff_hidden_dim),
                nn.ReLU(),
                nn.Linear(ff_hidden_dim, input_dim),
            )
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            attn_output, _ = self.self_attn(x, x, x)
            x = self.norm1(x + self.dropout(attn_output))
            x = self.norm2(x + self.dropout(self.self_ff(x)))
            return x


    class _Conv1dLayer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            kernel_size: int,
            dropout: float,
        ):
            super().__init__()
            self.conv1d = nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 1)
            x = self.relu(self.conv1d(x))
            x = self.dropout(x)
            x = x.permute(0, 2, 1)
            return x


    class _TransformerCnn(nn.Module):
        def __init__(
            self,
            response_dim: int,
            env_dim: int,
            trans_dim: int,
            num_heads: int,
            ff_hidden_dim: int,
            conv_hidden_dim: int,
            kernel_size: int,
            dropout: float,
            n_steps: int,
            lag: int,
            m: int,
        ):
            super().__init__()
            self.response_dim = response_dim
            self.n_steps = n_steps
            self.transformer = _TransformerEncoderLayer(
                input_dim=trans_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
            )
            self.conv_response = _Conv1dLayer(
                response_dim, conv_hidden_dim, kernel_size, dropout
            )
            self.conv_env = _Conv1dLayer(env_dim, conv_hidden_dim, kernel_size, dropout)
            self.conv_trans = _Conv1dLayer(
                trans_dim, conv_hidden_dim, kernel_size, dropout
            )
            self.final_conv = _Conv1dLayer(
                conv_hidden_dim, response_dim, kernel_size, dropout
            )
            fc_input_dim = response_dim * (lag + m * 2)
            self.fc = nn.Linear(fc_input_dim, response_dim * n_steps)

        def forward(
            self,
            x_response: torch.Tensor,
            x_env: torch.Tensor,
            x_cat: torch.Tensor,
        ) -> torch.Tensor:
            x_cat = self.transformer(x_cat)
            x_response_conv = self.conv_response(x_response)
            x_env_conv = self.conv_env(x_env)
            x_cat_conv = self.conv_trans(x_cat)
            x_concat = torch.cat([x_response_conv, x_env_conv, x_cat_conv], dim=1)
            x_final_conv = self.final_conv(x_concat)
            x_final_flat = x_final_conv.reshape(x_final_conv.size(0), -1)
            out = self.fc(x_final_flat)
            return out.view(x_final_conv.size(0), -1, self.response_dim)


# 默认路径（torch 有无均可使用）
_DEFAULT_MODEL_DIR = os.path.join(
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)
_DEFAULT_MODEL_CONFIG = {
    "response_dim": 3,
    "env_dim": 2,
    "trans_dim": 15,
    "num_heads": 3,
    "ff_hidden_dim": 128,
    "conv_hidden_dim": 96,
    "kernel_size": 3,
    "dropout": 0.25,
    "n_steps": 6,
    "lag": 80,
    "m": 30,
}


if _TORCH_AVAILABLE:
    # Transformer-CNN 适配器（.pth + scaler）
    class TransformerCNNAdapter(ModelAdapter):
        """
        可插拔 Transformer-CNN 裂纹预测适配器。
        支持 predict(x) 其中 x 为 (x_response, x_env, x_cat) 三元组；
        或 predict(x) 单数组（仅当后续扩展为单输入时使用）。
        """

        def __init__(
            self,
            model_path: str | None = None,
            scaler_all_path: str | None = None,
            scaler_response_path: str | None = None,
            model_config: Dict | None = None,
            device: str | torch.device | None = None,
        ):
            if model_path is None:
                model_path = os.path.join(_DEFAULT_MODEL_DIR, "best_crack_model.pth")
            if scaler_all_path is None:
                scaler_all_path = os.path.join(_DEFAULT_MODEL_DIR, "scaler_all.pkl")
            if scaler_response_path is None:
                scaler_response_path = os.path.join(
                    _DEFAULT_MODEL_DIR, "scaler_response.pkl"
                )
            cfg = model_config or _DEFAULT_MODEL_CONFIG
            self._config = cfg
            self._device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            # 构建模型并加载权重（.pth 为 state_dict）
            self.model = _TransformerCnn(
                response_dim=cfg["response_dim"],
                env_dim=cfg["env_dim"],
                trans_dim=cfg["trans_dim"],
                num_heads=cfg["num_heads"],
                ff_hidden_dim=cfg["ff_hidden_dim"],
                conv_hidden_dim=cfg["conv_hidden_dim"],
                kernel_size=cfg["kernel_size"],
                dropout=cfg["dropout"],
                n_steps=cfg["n_steps"],
                lag=cfg["lag"],
                m=cfg["m"],
            )
            state = torch.load(model_path, map_location=self._device, weights_only=False)
            self.model.load_state_dict(state)
            self.model.to(self._device)
            self.model.eval()

            with open(scaler_all_path, "rb") as f:
                self.scaler_all = pickle.load(f)
            with open(scaler_response_path, "rb") as f:
                self.scaler_response = pickle.load(f)

        def predict(
            self,
            x: Union[
                np.ndarray,
                Tuple[np.ndarray, np.ndarray, np.ndarray],
            ],
        ) -> np.ndarray:
            """
            推理。
            x: 可为单个 np.ndarray（占位，供其他单输入模型），
               或 (x_response, x_env, x_cat) 三元组，形状分别为
               [batch, m, response_dim], [batch, lag, env_dim], [batch, m, trans_dim]。
            返回: [batch, n_steps, response_dim]，已做 scaler_response 逆变换。
            """
            if isinstance(x, (tuple, list)) and len(x) == 3:
                return self._predict_three(x[0], x[1], x[2])
            if isinstance(x, np.ndarray):
                raise NotImplementedError(
                    "TransformerCNNAdapter 请使用 predict((x_response, x_env, x_cat)) 三输入"
                )
            raise TypeError("x 应为 np.ndarray 或 (x_response, x_env, x_cat) 三元组")

        def _predict_three(
            self,
            x_response: np.ndarray,
            x_env: np.ndarray,
            x_cat: np.ndarray,
        ) -> np.ndarray:
            """三输入推理并逆变换."""
            t_res = torch.from_numpy(x_response).float().to(self._device)
            t_env = torch.from_numpy(x_env).float().to(self._device)
            t_cat = torch.from_numpy(x_cat).float().to(self._device)
            with torch.no_grad():
                out = self.model(t_res, t_env, t_cat)
            out_np = out.cpu().numpy()
            batch, n_steps, rd = out_np.shape
            out_flat = out_np.reshape(-1, rd)
            out_orig = self.scaler_response.inverse_transform(out_flat)
            return out_orig.reshape(batch, n_steps, rd)


# ---------------------------------------------------------------------------
# 配置切换（prediction_service.py 等处使用）；tf28 在 tf28_adapter 中注册
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {}
if _TORCH_AVAILABLE:
    MODEL_REGISTRY["transformer_cnn"] = TransformerCNNAdapter


def get_adapter(
    name: str = "transformer_cnn",
    model_path: str | None = None,
    scaler_all_path: str | None = None,
    scaler_response_path: str | None = None,
    **kwargs,
) -> ModelAdapter:
    """根据名称从 REGISTRY 获取适配器实例。"""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型: {name}，可选: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[name]
    return cls(
        model_path=model_path,
        scaler_all_path=scaler_all_path,
        scaler_response_path=scaler_response_path,
        **kwargs,
    )
