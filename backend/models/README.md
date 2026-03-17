# 多源传感器 Transformer-CNN 模型

支持可接入多源传感器，用户可自行配置接入设备；提供 Transformer-CNN 时序预测模型的**参考实现**（网络结构 + 训练脚本 + 适配器）。**当前随仓库提供的预训练权重（best_crack_model.pth 及两个 scaler）仅针对测缝计（裂缝）训练，仅适用于裂缝预测场景，不适用于水准仪、测斜、水位计等其他传感器类型**；其他传感器需自行准备数据并训练后使用。若仓库内已包含上述权重文件，则测缝计内置模型可开箱使用；否则需先训练并放置权重后再使用。

## 内置模型说明（重要）

- **适用范围**：当前内置预训练权重**仅针对测缝计（裂缝）训练**，仅适用于裂缝预测，**不适用于水准仪、测斜、水位计等其他传感器**；其他传感器类型需使用本目录训练脚本自行训练并注册新模型。
- **预训练权重**：若仓库内已包含 `best_crack_model.pth`、`scaler_all.pkl`、`scaler_response.pkl`（位于 `backend/models/`），则测缝计可无需自行训练即使用内置 Transformer-CNN 预测；若未包含或已删除上述文件，则不存在“开箱即用”的预训练模型。
- **无权重时的行为**：`model_registry.json` 中注册的 `transformer_cnn` 条目其 `path` 指向的权重文件若不存在，系统会**静默使用 MockAdapter**：界面仍显示“Transformer-CNN”可选，但预测结果为简单外推占位，并非真实模型推理。
- **自行训练后使用**：使用本目录下的 `train.py` 训练得到 `.pth` 与 `.pkl` 后，将它们放到 `model_registry.json` 中配置的路径（默认即 `backend/models/`）。针对其他传感器类型训练时，建议在 `user_models/` 下新建模型目录并在注册表中添加新条目，与测缝计内置模型区分。放置后重启后端或调用 `DELETE /api/models/{model_id}/cache` 清缓存，即可加载真实权重。

## 可插拔预测模型方案

平台支持通过 **ULDM（通用逻辑数据模型）+ 模型适配器** 实现可插拔预测：统一输入语义、标准化输出（带时间戳的未来读数序列）、元数据驱动注册。完整设计见：**项目根目录 `docs/可插拔预测模型修改方案.md`**。元数据示例位于 `model_meta/` 目录。

## 目录结构

| 文件 | 说明 |
|------|------|
| `model_registry.json` | 模型注册表，内置 + 用户模型均在此配置 |
| `model_meta/` | 模型元数据（YAML），用于能力声明与注册，见 model_meta/README.md |
| `config.py` | 多源传感器配置与模型超参数 |
| `model_config.json` | 用户可编辑的配置文件 |
| `transformer_cnn.py` | 模型架构（Transformer + CNN） |
| `dataset.py` | 从分类型 CSV 加载、合并数据集 |
| `train.py` | 统一训练脚本 |
| `best_crack_model.pth` | 训练后权重（训练生成） |
| `scaler_all.pkl` | 全特征归一化器（训练生成） |
| `scaler_response.pkl` | 预测目标归一化器（训练生成） |
| `user_models/` | 用户自定义模型目录，见 user_models/README.md |

## 样例数据 (backend/sample_data)

每个 CSV 文件对应**一种传感器类型**，可含多个测点，同类型多传感器展示在同一图表中：

| 文件 | 说明 | 示例列 |
|------|------|--------|
| `settlement.csv` | 水准仪 | time, settlement_1, settlement_2, settlement_3, settlement_4 |
| `track.csv` | 测缝计 | time, crack_1, crack_2, crack_3 |
| `tilt_x.csv` | 测斜-X | time, tilt_x_1, tilt_x_2, tilt_x_3, tilt_x_4 |
| `water_level.csv` | 水位计 | time, water_level |
| `training_data.csv` | 已合并的完整训练数据（优先使用） | time + 所有传感器列 |

## 配置 (model_config.json)

用户可自行编辑 `model_config.json` 以：

1. **新增传感器类型**：添加 CSV 到 `sample_data/`，在 `sensor_types` 中配置
2. **指定角色**：`response`（预测目标）、`env`（环境）、`aux`（辅助）
3. **调整模型超参数**：m、n、lag、维度等

```json
{
  "sensor_types": [
    { "key": "crack", "file": "track.csv", "label": "测缝计", "role": "response", "channels": ["crack_1","crack_2","crack_3"] },
    { "key": "water_level", "file": "water_level.csv", "label": "水位计", "role": "env", "channels": ["water_level"] }
  ],
  "model": { "m": 30, "n": 6, "lag": 80, "response_dim": 3 }
}
```

## 训练

```bash
cd backend
python -m models.train
```

或指定配置：

```bash
MODEL_CONFIG=path/to/model_config.json python -m models.train
```

训练优先使用 `sample_data/training_data.csv`；若不存在，则从各分类型 CSV 按时间合并。

## 推理

模型适配器（`app/adapters/model_adapter.py`）根据 `model_registry.json` 中的 `path` 加载权重：若该路径下存在 `.pth` 文件，则加载 TransformerCNNAdapter 进行真实推理；**若不存在，则自动使用 MockAdapter 占位**（不报错，预测为简单外推）。用户训练完成并将 `.pth`、`scaler_all.pkl`、`scaler_response.pkl` 放到约定路径后，即可获得真实内置模型效果。
