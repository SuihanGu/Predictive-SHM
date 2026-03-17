# 可插拔模型适配器（步骤 1.1）
# MODEL_REGISTRY 用于在 prediction_service 等处切换模型
from . import model_adapter
from . import tf28_adapter  # 若已装 tensorflow，会注册 "tf28"
