# backend/app/services/config_loader.py
# 监控页配置加载：传感器列表、模型列表，支持用户通过 config 文件扩展

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# 配置路径：backend/config/monitor_config.json（支持多种解析方式）
def _resolve_config_path() -> Path:
    base = Path(__file__).resolve().parents[2]  # backend/
    p = base / "config" / "monitor_config.json"
    if p.exists():
        return p
    # 备选：从 cwd 的 backend/config 查找
    cwd_config = Path.cwd() / "config" / "monitor_config.json"
    if cwd_config.exists():
        return cwd_config
    return base / "config" / "monitor_config.json"


_sensors_cache: Optional[List[Dict[str, Any]]] = None
_models_cache: Optional[List[Dict[str, Any]]] = None


_DEFAULT_SENSORS = [
    {"key": "crack", "label": "Crack meter", "unit": "mm", "data_key": "crack", "default_static_threshold": 0.8, "default_residual_threshold": 0.1, "step": 0.1, "precision": 2, "full_width": True},
    {"key": "tilt_x", "label": "Tilt probe (X)", "unit": "°", "data_key": "tilt_x", "default_static_threshold": 0.5, "default_residual_threshold": 0.05, "step": 0.1, "precision": 2, "full_width": False},
    {"key": "tilt_y", "label": "Tilt probe (Y)", "unit": "°", "data_key": "tilt_y", "default_static_threshold": 0.5, "default_residual_threshold": 0.05, "step": 0.1, "precision": 2, "full_width": False},
    {"key": "settlement", "label": "Settlement", "unit": "mm", "data_key": "settlement", "default_static_threshold": 5.0, "default_residual_threshold": 0.5, "step": 0.5, "precision": 2, "full_width": False},
    {"key": "water_level", "label": "Water level", "unit": "mm", "data_key": "water_level", "default_static_threshold": 100, "default_residual_threshold": 10, "step": 1, "precision": 1, "full_width": False},
]
_DEFAULT_MODELS = [{"name": "transformer_cnn", "label": "Transformer-CNN", "description": "Crack forecasting (PyTorch)"}]


def _load_json() -> Dict[str, Any]:
    """加载配置文件，文件不存在或解析失败则返回默认结构"""
    path = _resolve_config_path()
    if not path.exists():
        return {"sensors": _DEFAULT_SENSORS, "models": _DEFAULT_MODELS}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("sensors"):
            data["sensors"] = _DEFAULT_SENSORS
        if not data.get("models"):
            data["models"] = _DEFAULT_MODELS
        return data
    except Exception:
        return {"sensors": _DEFAULT_SENSORS, "models": _DEFAULT_MODELS}


def _reload():
    """重新加载配置（修改 config 后需调用或重启服务生效）"""
    global _sensors_cache, _models_cache
    data = _load_json()
    _sensors_cache = data.get("sensors") or []
    if not _sensors_cache:
        _sensors_cache = [_DEFAULT_SENSORS[0]]  # 无传感器时使用测缝计
    _models_cache = data.get("models") or []


def get_sensors() -> List[Dict[str, Any]]:
    """返回传感器配置列表，用于前端动态渲染图表"""
    global _sensors_cache
    if _sensors_cache is None:
        _reload()
    return _sensors_cache or []


def get_models() -> List[Dict[str, Any]]:
    """返回模型配置列表，用于前端下拉选择；可与 MODEL_REGISTRY 合并过滤"""
    global _models_cache
    if _models_cache is None:
        _reload()
    return _models_cache or []


def get_full_config() -> Dict[str, Any]:
    """返回完整配置（sensors + models），供 GET /api/config 使用"""
    return {
        "sensors": get_sensors(),
        "models": get_models(),
    }
