#!/usr/bin/env python3
"""
修复远程版本的特征提取格式问题
"""

import os
import sys
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

def convert_case_features_to_slide_features(input_dir, output_dir, csv_path):
    """
    将按case_id保存的特征文件转换为按slide_id保存的特征文件
    
    Args:
        input_dir: 输入目录（按case_id保存的特征文件）
        output_dir: 输出目录（按slide_id保存的特征文件）
        csv_path: CSV文件路径，包含slide_id和case_id的映射
    """
    print("开始转换特征文件格式...")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"读取CSV文件: {len(df)} 行数据")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 按case_id分组
    case_groups = df.groupby('case_id')
    print(f"找到 {len(case_groups)} 个不同的case")
    
    converted_count = 0
    missing_count = 0
    
    for case_id, group in tqdm(case_groups, desc="转换特征文件"):
        # 输入特征文件路径
        input_feature_path = os.path.join(input_dir, f"{case_id}.h5")
        
        if not os.path.exists(input_feature_path):
            print(f"警告: 特征文件不存在: {input_feature_path}")
            missing_count += 1
            continue
        
        try:
            # 读取case级别的特征文件
            with h5py.File(input_feature_path, 'r') as f:
                features = f['features'][:]
                coords = f['coords'][:]
                
            # 为该case的每个slide_id创建单独的特征文件
            for idx, (_, row) in enumerate(group.iterrows()):
                slide_id = row['slide_id']
                
                # 构建输出路径（使用slide_id作为文件名）
                # 需要处理slide_id中的路径分隔符
                safe_slide_id = slide_id.replace('/', '_').replace('\\', '_')
                output_feature_path = os.path.join(output_dir, f"{safe_slide_id}.h5")
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_feature_path), exist_ok=True)
                
                # 保存单个slide的特征
                with h5py.File(output_feature_path, 'w') as f:
                    if idx < len(features):
                        # 保存对应的特征
                        f.create_dataset('features', data=features[idx:idx+1])
                        f.create_dataset('coords', data=coords[idx:idx+1])
                    else:
                        # 如果索引超出范围，使用第一个特征
                        f.create_dataset('features', data=features[0:1])
                        f.create_dataset('coords', data=coords[0:1])
                
                converted_count += 1
                
        except Exception as e:
            print(f"处理 {case_id} 时出错: {e}")
            missing_count += 1
    
    print(f"\n转换完成:")
    print(f"成功转换: {converted_count} 个特征文件")
    print(f"缺失/错误: {missing_count} 个特征文件")
    print(f"输出目录: {output_dir}")

def create_slide_based_features_directly(csv_path, mura_root, output_dir, model_name='clip_RN50'):
    """
    直接重新提取特征，按slide_id保存
    """
    print("重新提取特征，按slide_id保存...")
    
    # 这里可以调用修正后的特征提取脚本
    extract_cmd = f"""
python feature_extraction/extract_features.py \\
    --data_root {mura_root} \\
    --csv_path {csv_path} \\
    --output_dir {output_dir} \\
    --model_name {model_name} \\
    --batch_size 16 \\
    --save_by_slide_id
"""
    
    print("建议运行以下命令重新提取特征:")
    print(extract_cmd)

def main():
    """主函数"""
    # 配置路径
    remote_dir = "ViLa-MURA远程"
    input_dir = os.path.join(remote_dir, "processed_data/low_res_features")
    output_dir = os.path.join(remote_dir, "processed_data/low_res_features_fixed")
    csv_path = os.path.join(remote_dir, "dataset_csv/mura_abnormality_detection.csv")
    
    print("=== 远程版本特征提取格式修复 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"CSV文件: {csv_path}")
    
    # 检查文件是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV文件不存在: {csv_path}")
        return
    
    # 方案1: 转换现有特征文件
    print("\n方案1: 转换现有特征文件格式")
    try:
        convert_case_features_to_slide_features(input_dir, output_dir, csv_path)
        print("✅ 特征文件格式转换完成")
        
        print(f"\n请将训练命令中的特征目录改为:")
        print(f"--data_folder_s low_res_features_fixed")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        
        # 方案2: 重新提取特征
        print("\n方案2: 重新提取特征（推荐）")
        create_slide_based_features_directly(
            csv_path, 
            "../MURA-v1.1", 
            os.path.join(remote_dir, "processed_data/low_res_features_slide_based")
        )

if __name__ == "__main__":
    main()