from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.data import router as data_router
from app.routers.predict import router as predict_router
from app.routers.alerts import router as alerts_router
from app.routers.models import router as models_router
from app.routers.config_router import router as config_router

app = FastAPI(
    title="Predictive-SHM",
    description="轻量化、可扩展的结构健康监测与多步预测平台",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS（前端 Vite 默认端口 3000）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境改为 ["http://localhost:3000", "https://your-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router, prefix="/api", tags=["数据接入"])
app.include_router(predict_router, prefix="/api", tags=["预测模型"])
app.include_router(alerts_router, prefix="/api", tags=["预警配置"])
app.include_router(models_router, prefix="/api", tags=["模型管理"])
app.include_router(config_router, prefix="/api", tags=["配置"])

@app.get("/")
async def root():
    return {
        "message": "Predictive-SHM FastAPI 服务已启动",
        "docs": "/docs",
    }