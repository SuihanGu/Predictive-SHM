# -*- coding: utf-8 -*-
"""
通过 HTTP 预测接口测试是否调用了真实模型。
依赖：后端已启动（且使用 tf28 等有 torch 的环境），且已返回 adapter_type 字段。

用法：
  python scripts/check_predict_api.py
  python scripts/check_predict_api.py --base http://localhost:4999
"""
import os
import sys
import json
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_SCRIPT_DIR)
os.chdir(_BACKEND)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def build_sample_history():
    cols = (
        [f"settlement_{i+1}" for i in range(4)]
        + [f"crack_{i+1}" for i in range(3)]
        + [f"tilt_x_{i+1}" for i in range(4)]
        + [f"tilt_y_{i+1}" for i in range(4)]
        + ["water_level", "temperature"]
    )
    base_ts = 1700000000
    history_data = []
    for i in range(100):
        row = {"timestamp": base_ts + i * 600}
        for j, c in enumerate(cols):
            row[c] = 0.1 + 0.02 * (i % 10) + 0.01 * (j % 3)
        history_data.append(row)
    return history_data


def main():
    parser = argparse.ArgumentParser(description="Test /api/predict uses real model")
    parser.add_argument("--base", default="http://localhost:5173", help="Base URL (e.g. http://localhost:5173 or http://localhost:4999)")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    url = f"{base}/api/predict"

    try:
        import urllib.request
        body = json.dumps({
            "history_data": build_sample_history(),
            "model_name": "transformer_cnn",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"请求失败: {e}")
        print("请确认后端已启动（且若用 Vite 代理，前端需在运行）。可指定 --base 如 http://localhost:4999")
        return 1

    adapter_type = data.get("adapter_type") or data.get("model_used")
    model_used = data.get("model_used", "")
    shape = data.get("shape", [])
    prediction = data.get("prediction", {})

    print("API 响应摘要:")
    print(f"  model_used: {model_used}")
    print(f"  adapter_type: {adapter_type}")
    print(f"  shape: {shape}")
    if prediction:
        print(f"  prediction keys: {list(prediction.keys())}")
        if prediction.get("readings"):
            print(f"  readings rows: {len(prediction['readings'])}")

    if adapter_type == "TransformerCNNAdapter":
        print("\n[通过] 预测接口已调用真实 Transformer-CNN 模型（非 MockAdapter）")
        return 0
    if adapter_type == "MockAdapter":
        print("\n[未通过] 当前使用的是 MockAdapter 占位，未加载真实权重。请用 tf28 环境启动后端并确认 models/ 下存在 .pth 与 scaler 文件。")
        return 1
    print("\n[未知] 无法从响应判断是否真实模型；若后端已更新，响应中应包含 adapter_type 字段。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
