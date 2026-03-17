# -*- coding: utf-8 -*-
"""
多源传感器数据集：从 sample_data 分类型 CSV 加载并合并
每个 CSV 文件为一种传感器类型，可含多个测点，合并后供模型训练使用。
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from .config import ModelConfig, SAMPLE_DATA_DIR, DEFAULT_CONFIG_PATH


def load_sensor_csv(base_dir: str, filename: str, time_col: str = "time") -> Optional[pd.DataFrame]:
    """加载单种传感器 CSV，首列为时间，其余为数据列"""
    path = os.path.join(base_dir, filename)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.lower() in ("time", "timestamp") or df.columns.get_loc(c) == 0]
    if not cols:
        df = df.rename(columns={df.columns[0]: time_col})
    else:
        tc = cols[0]
        if tc != time_col and "time" in tc.lower():
            df = df.rename(columns={tc: time_col})
    df[time_col] = pd.to_datetime(df[time_col])
    return df


def merge_sensor_data(
    base_dir: str,
    config: ModelConfig,
    time_col: str = "time",
) -> pd.DataFrame:
    """
    按配置合并 sample_data 下各传感器 CSV。
    每个 CSV 对应一种传感器类型，可含多个通道（如 tilt_x_1, tilt_x_2...），
    同一类型的多个传感器将显示在同一图表中。
    """
    dfs: List[pd.DataFrame] = []
    for st in config.sensor_types:
        df = load_sensor_csv(base_dir, st.file, time_col)
        if df is not None:
            data_cols = [c for c in df.columns if c != time_col]
            if st.channels:
                data_cols = [c for c in st.channels if c in df.columns]
                if not data_cols:
                    data_cols = [c for c in df.columns if c != time_col]
            use_cols = [time_col] + data_cols
            dfs.append(df[use_cols].copy())
        # 文件不存在则跳过，合并后通过列序补 0

    if not dfs:
        raise FileNotFoundError(f"未在 {base_dir} 找到任何传感器 CSV")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=time_col, how="outer")
    merged = merged.sort_values(time_col).drop_duplicates(subset=[time_col])
    merged = merged.set_index(time_col)
    merged = merged.resample("10min").mean()
    merged = merged.interpolate(method="linear").ffill().bfill()

    # 确保列顺序与 config 一致
    order = config.columns_order()
    for c in order:
        if c not in merged.columns:
            merged[c] = 0.0
    cols = [c for c in order if c in merged.columns]
    return merged[cols].copy()


def load_training_data(
    base_dir: str = SAMPLE_DATA_DIR,
    config_path: str = DEFAULT_CONFIG_PATH,
    prefer_merged: bool = True,
) -> pd.DataFrame:
    """
    加载训练数据。优先级：
    1. training_data.csv（已合并的完整数据）
    2. 按 model_config 从分类型 CSV 合并
    """
    config = ModelConfig.from_json(config_path)
    merged_path = os.path.join(base_dir, "training_data.csv")

    if prefer_merged and os.path.isfile(merged_path):
        df = pd.read_csv(merged_path)
        time_col = "time" if "time" in df.columns else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()
        df = df.resample("10min").mean().interpolate(method="linear").ffill().bfill()
        order = config.columns_order()
        for c in order:
            if c not in df.columns:
                df[c] = 0.0
        cols = [c for c in order if c in df.columns]
        return df[cols].copy()

    return merge_sensor_data(base_dir, config)


def create_sequences(
    df: np.ndarray,
    config: ModelConfig,
    response_col_idx: List[int],
    env_col_idx: List[int],
    trans_col_idx: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建 (X_response, X_env, X_cat, y_response) 序列
    df: [T, total_features] 归一化后的数组
    """
    m, n, lag = config.m, config.n, config.lag
    T = len(df)
    if T < lag + m + n:
        raise ValueError(f"数据长度 {T} 不足，需要至少 lag+m+n={lag + m + n}")

    X_r, X_e, X_c, y_r = [], [], [], []
    for i in range(lag, T - m - n + 1):
        X_r.append(df[i : i + m, response_col_idx])
        X_e.append(df[i - lag + m : i + m, env_col_idx])
        X_c.append(df[i : i + m, trans_col_idx])
        y_r.append(df[i + m : i + m + n, response_col_idx])

    return (
        np.array(X_r, dtype=np.float32),
        np.array(X_e, dtype=np.float32),
        np.array(X_c, dtype=np.float32),
        np.array(y_r, dtype=np.float32),
    )


def get_column_indices(config: ModelConfig) -> Tuple[List[int], List[int], List[int]]:
    """根据 config 返回 response、env、trans 的列索引，保证维度与 model 一致"""
    order = config.columns_order()

    def idx(cols: List[str], target_dim: int) -> List[int]:
        indices = [order.index(c) for c in cols if c in order]
        # 不足时用 order 中未使用的列索引补足
        used = set(indices)
        for i in range(len(order)):
            if len(indices) >= target_dim:
                break
            if i not in used:
                indices.append(i)
                used.add(i)
        return indices[:target_dim]

    ri = idx(config.response_columns(), config.response_dim)
    ei = idx(config.env_columns(), config.env_dim)
    trans_cols = [c for st in config.sensor_types if st.role in ("response", "aux") for c in st.channels]
    trans_cols = [c for c in trans_cols if c in order]
    ti = idx(trans_cols, config.trans_dim)
    return ri, ei, ti
