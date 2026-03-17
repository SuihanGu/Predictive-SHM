# -*- coding: utf-8 -*-
"""
模型注册表：从 model_registry.json 加载配置，按需实例化适配器。
用户将训练好的模型放入 backend/models/user_models/{model_id}/ 即可通过配置注册。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Any, Optional

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REGISTRY_PATH = os.path.join(_BACKEND_ROOT, "models", "model_registry.json")
_ADAPTER_CACHE: Dict[str, Any] = {}


def _resolve_path(rel_path: str) -> str:
    """将相对路径解析为绝对路径（相对于 backend 根目录）"""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(_BACKEND_ROOT, rel_path)


def _load_meta_file(meta_path: str) -> Dict[str, Any]:
    """加载元数据文件（JSON），若路径无效或非 JSON 则返回空 dict"""
    if not meta_path or not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_registry() -> List[Dict[str, Any]]:
    """加载 model_registry.json，合并 meta_file 中的能力与配置，返回模型配置列表"""
    if not os.path.isfile(_REGISTRY_PATH):
        return []
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", [])
        result = []
        for m in models:
            cfg = dict(m)
            meta_file = cfg.get("meta_file")
            if meta_file:
                resolved = _resolve_path(meta_file)
                meta = _load_meta_file(resolved)
                if meta:
                    cfg.setdefault("capabilities", meta.get("capabilities", {}))
                    for k, v in meta.items():
                        if k not in cfg:
                            cfg[k] = v
            result.append(cfg)
        return result
    except Exception:
        return []


def get_adapter(model_id: str):
    """
    按 model_id 获取适配器实例，带缓存。
    若模型文件不存在则返回 MockAdapter。
    """
    if model_id in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[model_id]

    from app.adapters.model_adapter import (
        ModelAdapter,
        MockAdapter,
        TransformerCNNAdapter,
        get_onnx_adapter_class,
    )

    registry = load_registry()
    cfg = next((m for m in registry if m.get("id") == model_id), None)
    if not cfg:
        return MockAdapter()

    adapter_name = cfg.get("adapter", "")
    path = _resolve_path(cfg.get("path", ""))
    scaler_path = _resolve_path(cfg.get("scaler_path", "")) if cfg.get("scaler_path") else None
    response_scaler_path = _resolve_path(cfg.get("response_scaler_path", "")) if cfg.get("response_scaler_path") else None

    adapter = None
    if adapter_name == "TransformerCNNAdapter" and TransformerCNNAdapter is not None:
        if os.path.isfile(path):
            adapter = TransformerCNNAdapter(
                model_path=path,
                scaler_path=scaler_path or _resolve_path("models/scaler_all.pkl"),
                response_scaler_path=response_scaler_path or _resolve_path("models/scaler_response.pkl"),
            )
        else:
            adapter = MockAdapter()
    elif adapter_name == "ONNXAdapter":
        OnnxCls = get_onnx_adapter_class()
        if OnnxCls and os.path.isfile(path):
            meta = cfg.get("meta_path")
            meta_path = _resolve_path(meta) if meta else None
            adapter = OnnxCls(
                model_path=path,
                meta_path=meta_path,
                target_sensor=cfg.get("target_sensor", ""),
                output_dim=cfg.get("output_dim", 1),
                pred_steps=cfg.get("pred_steps", 6),
            )
        else:
            adapter = MockAdapter()
    else:
        adapter = MockAdapter()

    _ADAPTER_CACHE[model_id] = adapter
    return adapter


def list_models() -> List[Dict[str, Any]]:
    """返回所有已注册模型的元信息，含 capabilities，用于前端下拉与传感器映射"""
    registry = load_registry()
    result = []
    for m in registry:
        model_id = m.get("id", "")
        path = _resolve_path(m.get("path", ""))
        available = os.path.isfile(path)
        entry = {
            "id": model_id,
            "name": model_id,
            "label": m.get("label", model_id),
            "description": m.get("description", ""),
            "target_sensor": m.get("target_sensor", ""),
            "available": available,
        }
        if m.get("capabilities"):
            entry["capabilities"] = m["capabilities"]
        result.append(entry)
    return result


def clear_cache(model_id: Optional[str] = None):
    """清除适配器缓存，model_id 为 None 时清除全部"""
    global _ADAPTER_CACHE
    if model_id:
        _ADAPTER_CACHE.pop(model_id, None)
    else:
        _ADAPTER_CACHE.clear()
