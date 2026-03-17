"""
训练数据预处理脚本：处理传感器断电造成的长时间缺失值
专门用于准备Transformer-CNN模型的训练数据
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def find_continuous_data_segments(df, max_gap_minutes=60, min_segment_length=None):
    """
    找到连续的数据段（没有长时间缺失的段）
    
    参数:
        df: DataFrame，包含time列和数据列
        max_gap_minutes: 允许的最大时间间隔（分钟），超过此间隔视为数据中断
        min_segment_length: 最小段长度（样本数），如果为None则不限制
    
    返回:
        segments: 连续数据段的列表，每个段是一个DataFrame
    """
    # 确保time是datetime类型
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    
    # 计算时间间隔（分钟）
    time_diffs = df.index.to_series().diff().dt.total_seconds() / 60
    
    # 找到数据中断点（时间间隔超过max_gap_minutes的点）
    break_points = np.where(time_diffs > max_gap_minutes)[0]
    
    # 如果没有中断点，整个数据就是一个连续段
    if len(break_points) == 0:
        segments = [df.reset_index()]
        print(f"找到1个连续数据段，总长度: {len(df)}")
        return segments
    
    # 分割数据段
    segments = []
    start_idx = 0
    
    for break_idx in break_points:
        segment = df.iloc[start_idx:break_idx].reset_index()
        if len(segment) > 0:
            segments.append(segment)
        start_idx = break_idx
    
    # 添加最后一段
    if start_idx < len(df):
        segment = df.iloc[start_idx:].reset_index()
        if len(segment) > 0:
            segments.append(segment)
    
    # 过滤掉太短的段
    if min_segment_length is not None:
        segments = [seg for seg in segments if len(seg) >= min_segment_length]
    
    print(f"找到 {len(segments)} 个连续数据段")
    for i, seg in enumerate(segments):
        print(f"  段 {i+1}: 长度={len(seg)}, 开始时间={seg['time'].min()}, 结束时间={seg['time'].max()}")
    
    return segments

def remove_rows_with_missing_values(df, columns_to_check=None):
    """
    移除包含缺失值的行
    
    参数:
        df: DataFrame
        columns_to_check: 要检查的列，如果为None则检查所有列
    
    返回:
        cleaned_df: 清理后的DataFrame
        removed_count: 移除的行数
    """
    if columns_to_check is None:
        columns_to_check = df.columns
    
    # 检查指定列的缺失值
    mask = df[columns_to_check].isna().any(axis=1)
    removed_count = mask.sum()
    cleaned_df = df[~mask].copy()
    
    return cleaned_df, removed_count

def interpolate_missing_values(df, max_gap_minutes=60, method='time'):
    """
    对缺失值进行插值，但限制最大插值间隔
    
    参数:
        df: DataFrame，包含time列
        max_gap_minutes: 最大允许插值的时间间隔（分钟）
        method: 插值方法 ('time', 'linear', 'forward', 'backward')
    
    返回:
        filled_df: 填充后的DataFrame
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    
    # 计算时间间隔
    time_diffs = df.index.to_series().diff().dt.total_seconds() / 60
    
    # 对每列进行插值
    for col in df.columns:
        if df[col].isna().any():
            # 先进行插值
            if method == 'time':
                df[col] = df[col].interpolate(method='time', limit_direction='both')
            elif method == 'linear':
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            elif method == 'forward':
                df[col] = df[col].fillna(method='ffill')
            elif method == 'backward':
                df[col] = df[col].fillna(method='bfill')
            
            # 对于超过max_gap_minutes的间隔，清空插值结果
            if max_gap_minutes > 0:
                large_gaps = time_diffs > max_gap_minutes
                # 找到大间隔后的第一个值
                for i, gap in enumerate(large_gaps):
                    if gap and i < len(df):
                        idx = df.index[i]
                        # 如果这个位置原本是缺失值，清空插值结果
                        prev_idx = df.index[df.index < idx]
                        if len(prev_idx) > 0:
                            prev_val = df.loc[prev_idx[-1], col]
                            if pd.isna(prev_val):
                                df.loc[idx, col] = np.nan
    
    df = df.reset_index()
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def prepare_training_data(input_file, output_file, strategy='remove_segments', 
                          max_gap_minutes=60, min_segment_length=None, 
                          interpolation_method='time', required_columns=None):
    """
    准备训练数据
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        strategy: 处理策略
            - 'remove_segments': 删除包含长时间缺失的整个数据段
            - 'interpolate': 对缺失值进行插值（限制最大间隔）
            - 'remove_rows': 删除包含缺失值的行
            - 'hybrid': 混合策略：先删除长时间缺失段，再对短时间缺失进行插值
        max_gap_minutes: 最大允许插值的时间间隔（分钟）
        min_segment_length: 最小段长度（用于remove_segments策略）
        interpolation_method: 插值方法 ('time', 'linear', 'forward', 'backward')
        required_columns: 必须完整的列（如果为None，则检查所有数据列）
    """
    print("=" * 60)
    print("训练数据预处理")
    print("=" * 60)
    print()
    
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        return None
    
    # 读取数据
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"原始数据行数: {len(df)}")
    print(f"原始数据列数: {len(df.columns)}")
    print()
    
    # 确保time列存在
    if 'time' not in df.columns:
        print("错误: 数据中缺少'time'列")
        return None
    
    # 确定要检查的列
    if required_columns is None:
        required_columns = [col for col in df.columns if col != 'time']
    
    print(f"处理策略: {strategy}")
    print(f"最大允许间隔: {max_gap_minutes} 分钟")
    print(f"需要完整的列: {required_columns}")
    print()
    
    processed_df = None
    
    if strategy == 'remove_segments':
        # 策略1: 删除包含长时间缺失的整个数据段
        print("策略: 删除包含长时间缺失的整个数据段")
        print()
        
        # 先找到连续数据段
        segments = find_continuous_data_segments(df, max_gap_minutes=max_gap_minutes, 
                                                 min_segment_length=min_segment_length)
        
        # 对每个段，移除包含缺失值的行
        cleaned_segments = []
        total_removed = 0
        
        for i, segment in enumerate(segments):
            print(f"处理段 {i+1}/{len(segments)}...")
            cleaned_seg, removed = remove_rows_with_missing_values(segment, columns_to_check=required_columns)
            cleaned_segments.append(cleaned_seg)
            total_removed += removed
            print(f"  移除 {removed} 行包含缺失值的数据")
        
        # 合并所有清理后的段
        if cleaned_segments:
            processed_df = pd.concat(cleaned_segments, ignore_index=True)
            print()
            print(f"总共移除 {total_removed} 行数据")
            print(f"保留 {len(processed_df)} 行数据")
        else:
            print("警告: 所有数据段都被移除了")
            return None
    
    elif strategy == 'interpolate':
        # 策略2: 对缺失值进行插值
        print("策略: 对缺失值进行插值（限制最大间隔）")
        print()
        
        processed_df = interpolate_missing_values(df, max_gap_minutes=max_gap_minutes, 
                                                 method=interpolation_method)
        
        # 统计插值后的缺失值
        remaining_missing = processed_df[required_columns].isna().sum().sum()
        print(f"插值后仍有 {remaining_missing} 个缺失值")
        
        # 如果还有缺失值，可以选择删除这些行
        if remaining_missing > 0:
            print("移除仍包含缺失值的行...")
            processed_df, removed = remove_rows_with_missing_values(processed_df, columns_to_check=required_columns)
            print(f"移除 {removed} 行")
    
    elif strategy == 'remove_rows':
        # 策略3: 直接删除包含缺失值的行
        print("策略: 删除包含缺失值的行")
        print()
        
        processed_df, removed = remove_rows_with_missing_values(df, columns_to_check=required_columns)
        print(f"移除 {removed} 行数据")
        print(f"保留 {len(processed_df)} 行数据")
    
    elif strategy == 'hybrid':
        # 策略4: 混合策略
        print("策略: 混合策略（先删除长时间缺失段，再对短时间缺失进行插值）")
        print()
        
        # 步骤1: 找到连续数据段
        segments = find_continuous_data_segments(df, max_gap_minutes=max_gap_minutes, 
                                                 min_segment_length=min_segment_length)
        
        # 步骤2: 对每个段进行插值
        interpolated_segments = []
        for i, segment in enumerate(segments):
            print(f"处理段 {i+1}/{len(segments)}...")
            # 对短时间缺失进行插值（使用较小的max_gap_minutes）
            interpolated_seg = interpolate_missing_values(segment, max_gap_minutes=30, 
                                                         method=interpolation_method)
            interpolated_segments.append(interpolated_seg)
        
        # 步骤3: 合并并移除仍有缺失值的行
        if interpolated_segments:
            processed_df = pd.concat(interpolated_segments, ignore_index=True)
            processed_df, removed = remove_rows_with_missing_values(processed_df, columns_to_check=required_columns)
            print(f"移除 {removed} 行仍包含缺失值的数据")
            print(f"保留 {len(processed_df)} 行数据")
    
    else:
        print(f"错误: 未知的处理策略: {strategy}")
        return None
    
    # 保存处理后的数据
    if processed_df is not None and len(processed_df) > 0:
        processed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print()
        print("=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"输出文件: {output_file}")
        print(f"最终数据行数: {len(processed_df)}")
        print(f"数据保留率: {len(processed_df)/len(df)*100:.2f}%")
        
        # 检查数据连续性
        processed_df['time'] = pd.to_datetime(processed_df['time'])
        time_diffs = processed_df['time'].diff().dt.total_seconds() / 60
        max_diff = time_diffs.max()
        print(f"最大时间间隔: {max_diff:.1f} 分钟")
        print(f"平均时间间隔: {time_diffs.mean():.1f} 分钟")
        
        return processed_df
    else:
        print("错误: 处理后没有剩余数据")
        return None

def main():
    """主函数"""
    # 文件路径
    input_file = "data_export_processed/合并数据_填充.csv"  # 或使用合并数据.csv
    output_file = "data_export_processed/训练数据.csv"
    
    # 如果填充后的文件不存在，使用未填充的文件
    if not os.path.exists(input_file):
        input_file = "data_export_processed/合并数据.csv"
    
    # 处理策略选择
    # 'remove_segments': 推荐用于训练，删除包含长时间缺失的段，保证数据连续性
    # 'hybrid': 如果数据量不足，可以使用混合策略
    # 'interpolate': 如果必须保留所有数据，可以使用插值
    # 'remove_rows': 简单粗暴，直接删除缺失行
    
    strategy = 'remove_segments'  # 推荐策略
    max_gap_minutes = 60  # 最大允许间隔（分钟）
    min_segment_length = 120  # 最小段长度（样本数），用于训练至少需要 m + n + lag = 116 个样本
    
    # 根据训练代码，需要的数据列
    # 列顺序：time, settlement_1-4, crack_1-3, tilt_x_1-4, tilt_y_1-4, water_level, temperature
    required_columns = [
        'settlement_1', 'settlement_2', 'settlement_3', 'settlement_4',
        'crack_1', 'crack_2', 'crack_3',
        'tilt_x_1', 'tilt_y_1', 'tilt_x_2', 'tilt_y_2',
        'tilt_x_3', 'tilt_y_3', 'tilt_x_4', 'tilt_y_4',
        'water_level', 'temperature'
    ]
    
    print("训练数据预处理配置:")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  处理策略: {strategy}")
    print(f"  最大允许间隔: {max_gap_minutes} 分钟")
    print(f"  最小段长度: {min_segment_length} 个样本")
    print()
    
    # 处理数据
    processed_df = prepare_training_data(
        input_file=input_file,
        output_file=output_file,
        strategy=strategy,
        max_gap_minutes=max_gap_minutes,
        min_segment_length=min_segment_length,
        interpolation_method='time',
        required_columns=required_columns
    )
    
    if processed_df is not None:
        print()
        print("建议:")
        print("1. 检查输出文件，确保数据连续性")
        print("2. 如果数据量不足，可以尝试:")
        print("   - 降低 max_gap_minutes 参数")
        print("   - 降低 min_segment_length 参数")
        print("   - 使用 'hybrid' 策略")
        print("3. 如果数据量充足，当前策略最适合训练")

if __name__ == "__main__":
    main()




