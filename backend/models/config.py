# -*- coding: utf-8 -*-
"""
多源传感器配置与模型超参数
支持用户自行配置接入的传感器类型、预测目标、辅助输入等。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(MODELS_DIR)
SAMPLE_DATA_DIR = os.path.join(BACKEND_ROOT, "sample_data")
DEFAULT_CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")


@dataclass
class SensorTypeConfig:
    """单种传感器类型配置（对应一个 CSV 文件，可含多个测点）"""
    key: str                    # 唯一标识，与 CSV 文件名一致（不含.csv）
    file: str                   # CSV 文件名，如 tilt_x.csv
    label: str                  # 显示名称
    unit: str = ""
    channels: List[str] = field(default_factory=list)  # 数据列名，如 ["tilt_x_1","tilt_x_2",...]
    role: str = "aux"           # response|env|aux  预测目标|环境输入|辅助输入


@dataclass
class ModelConfig:
    """Transformer-CNN 模型与训练配置"""
    # 传感器类型与角色
    sensor_types: List[SensorTypeConfig] = field(default_factory=list)

    # 模型超参数
    response_dim: int = 3        # 预测目标维度（crack_1,2,3）
    env_dim: int = 2            # 环境输入维度（water_level, temperature）
    trans_dim: int = 15         # Transformer 输入维度（response + aux 等）
    num_heads: int = 3
    ff_hidden_dim: int = 128
    conv_hidden_dim: int = 96
    kernel_size: int = 3
    dropout: float = 0.25

    # 时序参数
    m: int = 30                 # 输入窗口长度
    n: int = 6                  # 预测步数
    lag: int = 80               # 环境滞后长度

    # 训练参数
    train_ratio: float = 0.8
    batch_size: int = 32
    num_epochs: int = 200
    patience: int = 20
    learning_rate: float = 0.0003
    weight_decay: float = 1e-5
    kfold_splits: int = 5

    @classmethod
    def from_json(cls, path: str = DEFAULT_CONFIG_PATH) -> "ModelConfig":
        """从 JSON 加载配置"""
        if not os.path.isfile(path):
            return cls.from_defaults()
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        sensors = [
            SensorTypeConfig(
                key=s["key"],
                file=s.get("file", f"{s['key']}.csv"),
                label=s.get("label", s["key"]),
                unit=s.get("unit", ""),
                channels=s.get("channels", []),
                role=s.get("role", "aux"),
            )
            for s in raw.get("sensor_types", [])
        ]
        hp = raw.get("model", {})
        return cls(
            sensor_types=sensors,
            response_dim=hp.get("response_dim", 3),
            env_dim=hp.get("env_dim", 2),
            trans_dim=hp.get("trans_dim", 15),
            num_heads=hp.get("num_heads", 3),
            ff_hidden_dim=hp.get("ff_hidden_dim", 128),
            conv_hidden_dim=hp.get("conv_hidden_dim", 96),
            kernel_size=hp.get("kernel_size", 3),
            dropout=hp.get("dropout", 0.25),
            m=hp.get("m", 30),
            n=hp.get("n", 6),
            lag=hp.get("lag", 80),
            train_ratio=hp.get("train_ratio", 0.8),
            batch_size=hp.get("batch_size", 32),
            num_epochs=hp.get("num_epochs", 200),
            patience=hp.get("patience", 20),
            learning_rate=hp.get("learning_rate", 0.0003),
            weight_decay=hp.get("weight_decay", 1e-5),
            kfold_splits=hp.get("kfold_splits", 5),
        )

    @classmethod
    def from_defaults(cls) -> "ModelConfig":
        """基于 sample_data 的默认配置（与原有 crack 预测一致）"""
        return cls(
            sensor_types=[
                SensorTypeConfig("settlement", "settlement.csv", "水准仪", "mm", ["settlement_1","settlement_2","settlement_3","settlement_4"], "aux"),
                SensorTypeConfig("crack", "track.csv", "测缝计", "mm", ["crack_1","crack_2","crack_3"], "response"),
                SensorTypeConfig("tilt_x", "tilt_x.csv", "测斜-X", "°", ["tilt_x_1","tilt_x_2","tilt_x_3","tilt_x_4"], "aux"),
                SensorTypeConfig("tilt_y", "tilt_y.csv", "测斜-Y", "°", ["tilt_y_1","tilt_y_2","tilt_y_3","tilt_y_4"], "aux"),
                SensorTypeConfig("water_level", "water_level.csv", "水位计", "mm", ["water_level"], "env"),
                SensorTypeConfig("temperature", "temperature.csv", "温度", "°C", ["temperature"], "env"),
            ],
            response_dim=3,
            env_dim=2,
            trans_dim=15,
        )

    def columns_order(self) -> List[str]:
        """模型期望的列顺序（与 data_processor / adapter 一致）"""
        order = []
        for st in self.sensor_types:
            if st.channels:
                order.extend(st.channels)
            else:
                # 若未指定，尝试常见命名
                order.extend([f"{st.key}_{i+1}" for i in range(4)] if st.role != "env" else [st.key])
        return order

    def response_columns(self) -> List[str]:
        return [c for st in self.sensor_types if st.role == "response" for c in st.channels]

    def env_columns(self) -> List[str]:
        return [c for st in self.sensor_types if st.role == "env" for c in st.channels]

    def save(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        """保存配置到 JSON"""
        data = {
            "sensor_types": [
                {
                    "key": s.key,
                    "file": s.file,
                    "label": s.label,
                    "unit": s.unit,
                    "channels": s.channels,
                    "role": s.role,
                }
                for s in self.sensor_types
            ],
            "model": {
                "response_dim": self.response_dim,
                "env_dim": self.env_dim,
                "trans_dim": self.trans_dim,
                "num_heads": self.num_heads,
                "ff_hidden_dim": self.ff_hidden_dim,
                "conv_hidden_dim": self.conv_hidden_dim,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "m": self.m,
                "n": self.n,
                "lag": self.lag,
                "train_ratio": self.train_ratio,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "patience": self.patience,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "kfold_splits": self.kfold_splits,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
