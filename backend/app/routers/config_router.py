# backend/app/routers/config_router.py
# 监控页配置 API：传感器、模型由 config 文件驱动

from fastapi import APIRouter
from app.services.config_loader import get_full_config

router = APIRouter()


@router.get("/config/monitor")
async def get_monitor_config():
    """
    返回监控页配置：传感器列表（表头、单位、阈值等）、模型列表。
    前端根据此配置动态渲染图表，用户可修改 backend/config/monitor_config.json 扩展传感器与模型。
    """
    return get_full_config()
