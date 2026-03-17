# -*- coding: utf-8 -*-
"""
验证内置测缝计模型是否接入系统可用：
1. 检查 transformer_cnn 加载的是真实适配器（非 MockAdapter）
2. 用样例历史数据调用预测，检查返回形状与数值

使用方式（需已安装 torch 的环境，如 tf28）：
  conda activate tf28
  cd backend
  python scripts/check_model_integration.py
"""
import os
import sys

# 在 backend 根目录运行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_SCRIPT_DIR)
os.chdir(_BACKEND)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

def main():
    from app.adapters.registry import get_adapter, list_models
    from app.adapters.base import MockAdapter
    from app.services.uldm_builder import build_uldm
    import numpy as np

    print("=" * 50)
    print("1. 模型列表与可用性")
    print("=" * 50)
    models = list_models()
    for m in models:
        print(f"  id={m['id']}, available={m['available']}, label={m.get('label','')}")
    if not models:
        print("  未找到注册模型，请检查 model_registry.json")
        return 1

    print("\n" + "=" * 50)
    print("2. 加载 transformer_cnn 适配器")
    print("=" * 50)
    # 先检查 TransformerCNNAdapter 是否可用（依赖 torch 与 models.transformer_cnn）
    try:
        import torch
        print("  torch:", torch.__version__)
    except Exception as e:
        print("  torch 未安装或导入失败:", e)
    try:
        import sys
        _b = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".")
        if _b not in sys.path:
            sys.path.insert(0, _b)
        from models.transformer_cnn import TransformerCnn
        print("  models.transformer_cnn: OK")
    except Exception as e:
        print("  models.transformer_cnn 导入失败:", e)
    try:
        from app.adapters.model_adapter import TransformerCNNAdapter as TCNN
        tcnn_available = TCNN is not None
    except Exception as e:
        tcnn_available = False
        print("  导入 TransformerCNNAdapter 异常:", e)
    if not tcnn_available:
        print("  -> 将使用 MockAdapter，无法加载真实权重")
    # 解析路径，与 registry 一致（脚本在 backend/scripts/ 下，backend = 上一级目录）
    _path = os.path.join(_BACKEND, "models", "best_crack_model.pth")
    print(f"  权重路径: {os.path.abspath(_path)}")
    print(f"  文件存在: {os.path.isfile(_path)}")

    adapter = get_adapter("transformer_cnn")
    is_mock = isinstance(adapter, MockAdapter)
    print(f"  适配器类型: {type(adapter).__name__}")
    print(f"  是否为 MockAdapter（占位）: {is_mock}")
    if is_mock:
        print("  [失败] 未加载真实权重，请确认 backend/models/ 下存在 best_crack_model.pth 及 scaler 文件")
        return 1
    print("  [通过] 已加载真实 TransformerCNN 适配器")

    print("\n" + "=" * 50)
    print("3. 构造样例历史数据并预测")
    print("=" * 50)
    # 至少 80 个时间步（与 model_config lag 一致），17 列
    cols = (
        [f"settlement_{i+1}" for i in range(4)]
        + [f"crack_{i+1}" for i in range(3)]
        + [f"tilt_x_{i+1}" for i in range(4)]
        + [f"tilt_y_{i+1}" for i in range(4)]
        + ["water_level", "temperature"]
    )
    base_ts = 1700000000  # 示例时间戳
    history_data = []
    for i in range(100):
        row = {"timestamp": base_ts + i * 600}
        for j, c in enumerate(cols):
            row[c] = 0.1 + 0.02 * (i % 10) + 0.01 * (j % 3)
        history_data.append(row)

    uldm = build_uldm(history_data)
    print(f"  ULDM: time_steps={uldm.time_index.size}, targets={uldm.targets.shape}, full_matrix={uldm.full_matrix.shape if uldm.full_matrix is not None else None}")

    model_input = adapter.from_uldm(uldm)
    raw_output = adapter.predict(model_input)
    raw_output = np.asarray(raw_output)
    print(f"  原始预测形状: {raw_output.shape}")

    from datetime import datetime, timedelta
    import pandas as pd
    base_time = pd.Timestamp(uldm.time_index[-1]).to_pydatetime()
    sensor_ids = ["crack_1", "crack_2", "crack_3"]
    standard = adapter.to_standard_output(raw_output, base_time, 10, sensor_ids)
    pred = np.asarray(standard.readings)
    print(f"  标准化输出: time_index 长度={len(standard.time_index)}, readings 形状={pred.shape}")
    print(f"  预测值（首步）: {pred[0].tolist()}")

    # 预期 (6, 3) 或 (pred_steps, output_dim)
    if pred.ndim < 2 or pred.shape[0] < 1 or pred.shape[1] < 1:
        print("  [失败] 预测维度异常")
        return 1
    print("  [通过] ULDM → 适配器 → StandardPrediction 流程正常")

    print("\n" + "=" * 50)
    print("4. 结论")
    print("=" * 50)
    print("  内置测缝计模型已成功接入系统，预测接口可用。")
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n[异常] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
