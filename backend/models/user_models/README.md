# 用户自定义模型目录

将训练好的模型放入此目录，并在 `model_registry.json` 中注册即可使用。

## 目录结构示例

```
user_models/
├── tilt_x_v1/           # 测斜预测模型
│   ├── model.onnx      # ONNX 模型文件
│   └── meta.json       # 可选：输入输出维度等
└── settlement_lstm/    # 水准仪 LSTM 模型
    ├── model.pth       # PyTorch 权重（需对应适配器）
    └── scaler.pkl
```

## 注册步骤

1. 在 `user_models/` 下创建子目录，如 `tilt_x_v1/`
2. 将模型文件放入该目录
3. 编辑 `backend/models/model_registry.json`，在 `models` 数组中添加：

```json
{
  "id": "tilt_x_v1",
  "type": "onnx",
  "label": "测斜预测 v1",
  "adapter": "ONNXAdapter",
  "path": "models/user_models/tilt_x_v1/model.onnx",
  "meta_path": "models/user_models/tilt_x_v1/meta.json",
  "target_sensor": "tilt_x",
  "output_dim": 4,
  "pred_steps": 6
}
```

4. 重启后端或调用 `DELETE /api/models/{model_id}/cache` 清除缓存

## 支持的适配器

| adapter | 文件格式 | 说明 |
|---------|----------|------|
| TransformerCNNAdapter | .pth + .pkl | 裂缝预测，需 scaler_all.pkl、scaler_response.pkl |
| ONNXAdapter | .onnx | 通用 ONNX，需安装 onnxruntime |
