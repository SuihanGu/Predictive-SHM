from fastapi import APIRouter, Body, Request
from typing import List, Dict, Any
import numpy as np
import logging
from app.services.alert_service import AlertService

router = APIRouter()
alert_service = AlertService()
logger = logging.getLogger(__name__)


@router.get("/alerts/thresholds")
async def get_thresholds():
    return alert_service.thresholds


@router.post("/alerts/thresholds")
async def set_thresholds(th: Dict[str, Any] = Body(...)):
    """支持 { key: { static, residual } } 或旧格式 { key: number }"""
    alert_service.set_thresholds(th)
    return {"status": "updated", "thresholds": alert_service.thresholds}


@router.post("/alerts/check")
async def check_alerts(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    history = body.get("history", [])
    prediction = body.get("prediction", [])
    sensor_keys = body.get("sensor_keys")
    h = np.array(history) if history else np.zeros((1, 60, 5))
    p = np.array(prediction)
    alerts = alert_service.check_alerts(h, p, sensor_keys)
    return {"alerts": alerts}