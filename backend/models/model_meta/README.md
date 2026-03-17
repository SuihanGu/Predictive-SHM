# 模型元数据目录

本目录存放各模型的**元数据文件**（YAML/JSON），用于可插拔预测模块的元数据驱动注册。  
设计说明见项目根目录下 `docs/可插拔预测模型修改方案.md`。

- `transformer_cnn.yaml`：内置 Transformer-CNN 裂缝预测模型的元数据示例。
- 新增模型可在 `model_registry.json` 中通过 `meta_file` 指向 `model_meta/<model_id>.yaml`，或在 `user_models/<model_id>/meta.yaml` 中放置。

元数据用于声明模型能力（最小时间步、最大预测步、是否支持不确定性等）与输入/输出需求，便于平台校验与前端展示，无需修改后端核心预测逻辑即可注册新模型。
