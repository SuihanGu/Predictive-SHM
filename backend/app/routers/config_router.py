# backend/app/routers/config_router.py
# 监控页配置 API：传感器、模型由 config 文件驱动

from fastapi import APIRouter
from app.services.config_loader import get_full_config

router = APIRouter()


@router.get("/config/monitor")
async def get_monitor_config():
    """
    返回监控页配置：传感器列表（表头、单位、阈值、full_width、show_forecast 等）、模型列表。
    仅 show_forecast 为 true 的传感器会展示预测曲线与模型下拉；见 monitor_config.json。
    """
    return get_full_config()
