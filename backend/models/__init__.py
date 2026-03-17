# -*- coding: utf-8 -*-
"""
backend/models - 多源传感器 Transformer-CNN 模型
支持用户配置接入的传感器类型、预测目标、辅助输入。
"""
from .config import ModelConfig, SensorTypeConfig, SAMPLE_DATA_DIR, DEFAULT_CONFIG_PATH
from .dataset import load_training_data, merge_sensor_data, create_sequences, get_column_indices

try:
    from .transformer_cnn import TransformerCnn, TransformerEncoderLayer, Conv1dLayer
except ImportError:
    TransformerCnn = TransformerEncoderLayer = Conv1dLayer = None

__all__ = [
    "ModelConfig",
    "SensorTypeConfig",
    "TransformerCnn",
    "TransformerEncoderLayer",
    "Conv1dLayer",
    "load_training_data",
    "merge_sensor_data",
    "create_sequences",
    "get_column_indices",
    "SAMPLE_DATA_DIR",
    "DEFAULT_CONFIG_PATH",
]
