"""
合并数据空值填充：处理传感器断电造成的缺失值
提供多种填充策略，保持数据准确性
"""
import csv
import os
import pandas as pd
import numpy as np
from datetime import datetime

def fill_missing_values(df, method='linear', max_gap_minutes=60, limit_direction='both'):
    """
    填充缺失值
    
    参数:
        df: DataFrame，包含时间列和其他数据列
        method: 填充方法
            - 'linear': 线性插值（推荐，适合传感器数据）
            - 'time': 时间加权线性插值（更准确，考虑时间间隔）
            - 'forward': 前向填充（使用前一个值）
            - 'backward': 后向填充（使用后一个值）
            - 'forward-backward': 先前向填充，再后向填充
            - 'polynomial': 多项式插值（需要数据足够）
        max_gap_minutes: 最大允许插值的时间间隔（分钟），超过此间隔不进行插值
        limit_direction: 插值方向 'both', 'forward', 'backward'
    """
    # 将time列转换为datetime类型
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    # 按时间排序
    df = df.sort_index()
    
    # 统计缺失值
    missing_stats = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_stats[col] = missing_count
    
    if missing_stats:
        print(f"缺失值统计:")
        for col, count in missing_stats.items():
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count} 个缺失值 ({percentage:.2f}%)")
        print()
    
    # 对每列进行填充
    filled_df = df.copy()
    
    for col in df.columns:
        if df[col].isna().any():
            original_values = df[col].copy()
            
            # 计算时间间隔（分钟）
            time_diffs_minutes = df.index.to_series().diff().dt.total_seconds() / 60
            
            if method == 'linear' or method == 'time':
                # 线性插值（时间加权）
                if method == 'time':
                    # 使用时间加权插值
                    filled_df[col] = df[col].interpolate(method='time', limit_direction=limit_direction)
                else:
                    # 普通线性插值
                    filled_df[col] = df[col].interpolate(method='linear', limit_direction=limit_direction)
                
                # 检查时间间隔，如果间隔过大，保持为空
                if max_gap_minutes > 0:
                    # 找到大间隔的位置
                    large_gaps = time_diffs_minutes > max_gap_minutes
                    
                    # 对于大间隔后的缺失值，如果前后都有值则保留插值，否则清空
                    for i, gap in enumerate(large_gaps):
                        if gap and i < len(df):
                            idx = df.index[i]
                            # 检查前后是否有有效值
                            prev_idx = df.index[df.index < idx]
                            next_idx = df.index[df.index > idx]
                            
                            if len(prev_idx) > 0 and len(next_idx) > 0:
                                prev_val = original_values.loc[prev_idx[-1]]
                                next_val = original_values.loc[next_idx[0]]
                                
                                # 如果前后都有值，保留插值；否则如果在间隔内，清空
                                if pd.isna(prev_val) or pd.isna(next_val):
                                    # 如果这个位置本身是缺失值，清空插值结果
                                    if pd.isna(original_values.loc[idx]):
                                        filled_df.loc[idx, col] = np.nan
                
            elif method == 'forward':
                # 前向填充
                filled_df[col] = df[col].fillna(method='ffill')
                
            elif method == 'backward':
                # 后向填充
                filled_df[col] = df[col].fillna(method='bfill')
                
            elif method == 'forward-backward':
                # 先前向填充，再后向填充
                filled_df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
            elif method == 'polynomial':
                # 多项式插值（需要至少有3个非空值）
                try:
                    filled_df[col] = df[col].interpolate(method='polynomial', order=2, limit_direction=limit_direction)
                except:
                    # 如果多项式插值失败，使用线性插值
                    print(f"  警告: {col} 多项式插值失败，改用线性插值")
                    filled_df[col] = df[col].interpolate(method='linear', limit_direction=limit_direction)
    
    # 重置索引，将time列恢复为普通列
    filled_df = filled_df.reset_index()
    filled_df['time'] = filled_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return filled_df, missing_stats

def process_merged_data(input_file, output_file, method='linear', max_gap_minutes=60):
    """
    处理合并数据，填充空值
    """
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        return
    
    print(f"正在处理文件: {input_file}")
    print(f"填充方法: {method}")
    print(f"最大插值间隔: {max_gap_minutes} 分钟")
    print()
    
    # 读取CSV文件
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"原始数据行数: {len(df)}")
    print(f"原始数据列数: {len(df.columns)}")
    print()
    
    # 填充缺失值
    filled_df, missing_stats = fill_missing_values(df.copy(), method=method, max_gap_minutes=max_gap_minutes)
    
    # 统计填充后的缺失值
    remaining_missing = {}
    for col in filled_df.columns:
        if col != 'time':
            missing_count = filled_df[col].isna().sum()
            if missing_count > 0:
                remaining_missing[col] = missing_count
    
    if remaining_missing:
        print(f"填充后仍有缺失值的列:")
        for col, count in remaining_missing.items():
            percentage = (count / len(filled_df)) * 100
            print(f"  {col}: {count} 个缺失值 ({percentage:.2f}%)")
        print()
    else:
        print("所有缺失值已成功填充！")
        print()
    
    # 保存填充后的数据
    filled_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"处理完成!")
    print(f"  输出文件: {output_file}")
    print(f"  填充后行数: {len(filled_df)}")
    print()
    
    return filled_df, missing_stats, remaining_missing

def main():
    """主函数"""
    print("=" * 60)
    print("合并数据空值填充处理")
    print("=" * 60)
    print()
    
    # 文件路径
    input_dir = "data_export_processed"
    input_file = os.path.join(input_dir, "合并数据.csv")
    output_file = os.path.join(input_dir, "合并数据_填充.csv")
    
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        return
    
    # 填充方法选择
    # 'linear': 线性插值（推荐，适合传感器数据）
    # 'time': 时间加权线性插值（更准确，考虑时间间隔）
    # 'forward': 前向填充（使用前一个值）
    # 'backward': 后向填充（使用后一个值）
    # 'forward-backward': 先前向后向后填充
    # 'polynomial': 多项式插值（需要数据足够）
    fill_method = 'time'  # 推荐使用 'time'，考虑时间间隔的线性插值
    max_gap_minutes = 60  # 最大允许插值的时间间隔（分钟），超过此间隔不进行插值
    
    print(f"填充策略说明:")
    print(f"  - 方法: {fill_method}")
    if fill_method == 'time':
        print(f"  - 时间加权线性插值: 考虑时间间隔的线性插值，最适合传感器数据")
    elif fill_method == 'linear':
        print(f"  - 线性插值: 在前后两个非空值之间线性插值，适合传感器数据")
    print(f"  - 最大插值间隔: {max_gap_minutes} 分钟")
    print(f"  - 如果断电时间超过 {max_gap_minutes} 分钟，该时间段不进行插值")
    print(f"  - 建议: 对于传感器数据，使用 'time' 方法最准确")
    print()
    
    # 处理数据
    filled_df, missing_stats, remaining_missing = process_merged_data(
        input_file, 
        output_file, 
        method=fill_method,
        max_gap_minutes=max_gap_minutes
    )
    
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print()
    print("建议:")
    print("1. 查看填充后的数据，检查填充效果")
    print("2. 如果断电时间过长（>60分钟），建议保持为空或手动处理")
    print("3. 可以根据实际情况调整 max_gap_minutes 参数")
    print("4. 也可以尝试其他填充方法（forward, backward等）")

if __name__ == "__main__":
    main()

