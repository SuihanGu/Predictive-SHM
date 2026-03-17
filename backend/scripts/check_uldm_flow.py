# -*- coding: utf-8 -*-
"""
验证 ULDM → 适配器 → StandardPrediction 流程（不依赖真实权重）。
无论是否安装 torch，均可运行以确认可插拔预测模块代码路径正确。
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_SCRIPT_DIR)
os.chdir(_BACKEND)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def main():
    from app.schemas.uldm import ULDM, StandardPrediction
    from app.services.uldm_builder import build_uldm
    from app.adapters.registry import get_adapter, load_registry
    import numpy as np

    print("=" * 50)
    print("1. load_registry 含 meta_file 合并")
    print("=" * 50)
    registry = load_registry()
    for m in registry:
        caps = m.get("capabilities", {})
        print(f"  id={m.get('id')}, capabilities={caps}")
    if registry and registry[0].get("capabilities"):
        print("  [通过] meta_file 合并 capabilities 成功")
    else:
        print("  [提示] 无 meta_file 或 capabilities 为空（可接受）")

    print("\n" + "=" * 50)
    print("2. build_uldm(history_data)")
    print("=" * 50)
    cols = (
        [f"settlement_{i+1}" for i in range(4)]
        + [f"crack_{i+1}" for i in range(3)]
        + [f"tilt_x_{i+1}" for i in range(4)]
        + [f"tilt_y_{i+1}" for i in range(4)]
        + ["water_level", "temperature"]
    )
    history_data = []
    for i in range(120):
        row = {"timestamp": 1700000000 + i * 600}
        for j, c in enumerate(cols):
            row[c] = 0.1 + 0.02 * (i % 10) + 0.01 * (j % 3)
        history_data.append(row)
    uldm = build_uldm(history_data)
    assert uldm.full_matrix is not None
    assert uldm.time_index.size > 0
    print(f"  time_index.size={uldm.time_index.size}, targets.shape={uldm.targets.shape}, full_matrix.shape={uldm.full_matrix.shape}")
    print("  [通过] ULDM 构建正常")

    print("\n" + "=" * 50)
    print("3. 适配器 from_uldm → predict → to_standard_output")
    print("=" * 50)
    adapter = get_adapter("transformer_cnn")
    model_input = adapter.from_uldm(uldm)
    raw_output = adapter.predict(model_input)
    raw_output = np.asarray(raw_output)
    import pandas as pd
    base_time = pd.Timestamp(uldm.time_index[-1]).to_pydatetime()
    sensor_ids = ["crack_1", "crack_2", "crack_3"]
    standard = adapter.to_standard_output(raw_output, base_time, 10, sensor_ids)
    assert hasattr(standard, "readings") and hasattr(standard, "time_index") and hasattr(standard, "sensor_ids")
    readings = np.asarray(standard.readings)
    print(f"  adapter_type={type(adapter).__name__}")
    print(f"  raw_output.shape={raw_output.shape}, standard.readings.shape={readings.shape}")
    print(f"  standard.time_index 长度={len(standard.time_index)}, sensor_ids={standard.sensor_ids}")
    d = standard.to_dict()
    assert "readings" in d and "time_index" in d and "sensor_ids" in d
    print("  [通过] StandardPrediction.to_dict() 可序列化")

    print("\n" + "=" * 50)
    print("4. get_capabilities")
    print("=" * 50)
    caps = adapter.get_capabilities()
    print(f"  {caps}")
    print("  [通过] 能力接口存在")

    print("\n" + "=" * 50)
    print("结论: ULDM 与可插拔预测流程验证通过")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n[异常] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
