# 监控配置

通过 `monitor_config.json` 手动添加传感器、调整表头与单位，并配置模型显示。

## 文件路径

`backend/config/monitor_config.json`

## 传感器配置 (sensors)

每个传感器包含：

| 字段 | 说明 |
|------|------|
| key | 唯一标识，用于数据字段、模型选择 |
| label | 表头/标题显示名称 |
| unit | 单位，显示在图表 Y 轴 |
| data_key | 历史数据中的字段名，默认与 key 相同 |
| default_static_threshold | 静态阈值：实测值超过此限值告警 |
| default_residual_threshold | 预测残差：\|预测−实测\| 超过此值告警 |
| default_threshold | （兼容旧版）等同于 default_residual_threshold |
| step | 阈值输入步长 |
| precision | 数值精度 |
| full_width | true 时图表独占一行，false 时两列并排 |

添加新传感器示例：

```json
{
  "key": "strain",
  "label": "应变计",
  "unit": "με",
  "data_key": "strain",
  "default_static_threshold": 500,
  "default_residual_threshold": 50,
  "step": 10,
  "precision": 0,
  "full_width": false
}
```

预警逻辑与预测模型**完全解耦**：仅根据 (history, prediction) 与阈值判断，不依赖具体模型。

新增传感器时，若无专属虚拟数据规则，会**自动复用测缝计**的生成模式。

## 模型配置 (models)

每个模型包含：

| 字段 | 说明 |
|------|------|
| name | 必须与 MODEL_REGISTRY 中注册名一致 |
| label | 前端下拉框显示名称 |
| description | 描述（可选） |

模型需在 `backend/app/adapters/model_adapter.py` 中注册实现，config 仅控制展示与可选范围。
