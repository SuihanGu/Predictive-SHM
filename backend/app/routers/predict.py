from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import traceback
import logging
import numpy as np
import pandas as pd
from app.adapters.registry import get_adapter, load_registry
from app.services.uldm_builder import build_uldm

router = APIRouter()
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    history_data: List[Dict]
    model_name: str = "transformer_cnn"


def _get_sensor_ids_for_model(model_id: str) -> List[str]:
    """从注册表或默认返回该模型预测目标传感器 ID 列表"""
    registry = load_registry()
    cfg = next((m for m in registry if m.get("id") == model_id), None)
    if not cfg:
        return ["crack_1", "crack_2", "crack_3"]
    target = cfg.get("target_sensor", "crack")
    output_dim = cfg.get("output_dim", 3)
    if target == "crack":
        return [f"crack_{i+1}" for i in range(output_dim)]
    return [f"{target}_{i+1}" for i in range(max(1, output_dim))]


@router.post("/predict")
async def predict(req: PredictRequest):
    """ULDM → 适配器 from_uldm → predict → to_standard_output，返回标准化预测结果"""
    try:
        uldm = build_uldm(req.history_data)
        if uldm.time_index.size == 0:
            raise ValueError("历史数据为空或重采样后无时间点")

        adapter = get_adapter(req.model_name)
        model_input = adapter.from_uldm(uldm)
        raw_output = adapter.predict(model_input)

        # 基准时间为最后一个采样时刻，步长 10 分钟
        last_ts = uldm.time_index[-1]
        base_time = pd.Timestamp(last_ts).to_pydatetime()
        step_minutes = 10
        sensor_ids = _get_sensor_ids_for_model(req.model_name)

        standard = adapter.to_standard_output(
            raw_output, base_time, step_minutes, sensor_ids
        )
        return {
            "model_used": req.model_name,
            "adapter_type": type(adapter).__name__,
            "base_time": base_time.isoformat(),
            "prediction": standard.to_dict(),
            "shape": list(np.asarray(standard.readings).shape),
        }
    except NotImplementedError as e:
        logger.exception("adapter does not support from_uldm")
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")