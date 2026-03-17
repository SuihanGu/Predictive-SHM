from fastapi import APIRouter
from app.adapters.registry import list_models, get_adapter, load_registry, clear_cache

router = APIRouter()


@router.get("/models/list")
async def api_list_models():
    """从 model_registry.json 返回已注册模型列表，含 target_sensor 供前端映射"""
    models = list_models()
    config_models = []
    try:
        from app.services.config_loader import get_models as get_config_models
        config_models = get_config_models() or []
    except Exception:
        pass
    return {
        "available_models": [m["id"] for m in models],
        "models": models,
        "models_config": config_models,
    }


@router.get("/models/{model_id}/meta")
async def get_model_meta(model_id: str):
    """返回指定模型的元信息"""
    adapter = get_adapter(model_id)
    meta = getattr(adapter, "get_meta", lambda: {})()
    registry = load_registry()
    cfg = next((m for m in registry if m.get("id") == model_id), None)
    if cfg:
        meta = {**cfg, **meta}
    return {"model_id": model_id, "meta": meta}


@router.post("/models/switch")
async def switch_model(model_name: str):
    """预加载模型到缓存，加速后续 predict 调用"""
    adapter = get_adapter(model_name)
    return {"status": "switched", "model": model_name}


@router.delete("/models/{model_id}/cache")
async def clear_model_cache(model_id: str):
    """清除指定模型的适配器缓存（用户更新模型文件后调用）"""
    clear_cache(model_id)
    return {"status": "cleared", "model_id": model_id}