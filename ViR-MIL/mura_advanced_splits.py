"""
MURA数据集高级分割策略
实现多种分割策略，包括按身体部位分层、患者级别分割和平衡采样
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA数据集高级分割')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/mura_abnormality_detection.csv',
                        help='MURA数据集CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='splits/task_mura_abnormality_detection_100',
                        help='输出分割文件目录')
    parser.add_argument('--k', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_frac', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--split_strategy', type=str, default='stratified_body_part',
                        choices=['simple', 'stratified_body_part', 'patient_level', 'balanced'],
                        help='分割策略')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化分割结果')
    parser.add_argument('--focus_body_part', type=str, default=None,
                        help='重点关注的身体部位，用于单部位实验')
    return parser.parse_args()

def extract_patient_id(df):
    """
    从slide_id提取患者ID
    
    参数:
        df: 数据集DataFrame
    
    返回:
        包含患者ID的DataFrame
    """
    df = df.copy()
    # 确保case_id列存在
    if 'case_id' not in df.columns:
        df['case_id'] = df['slide_id'].apply(lambda x: x.split('_')[0])
    return df

def create_simple_splits(df, output_dir, k=5, val_frac=0.1, test_frac=0.2, seed=42):
    """
    创建简单分割（仅按标签分层）
    
    参数:
        df: 数据集DataFrame
        output_dir: 输出目录
        k: 交叉验证折数
        val_frac: 验证集比例
        test_frac: 测试集比例
        seed: 随机种子
    """
    # 保留原始MURA的训练/验证分割
    train_df = df[df['split'] == 'train'].copy()
    valid_df = df[df['split'] == 'valid'].copy()
    
    # 为每个折创建分割
    for i in range(k):
        # 设置随机种子
        np.random.seed(seed + i)
        
        # 从训练集中分出测试集
        train_remain, test = train_test_split(train_df, test_size=test_frac, 
                                            stratify=train_df['label'], random_state=seed+i)
        
        # 从剩余训练集中分出验证集
        train, val = train_test_split(train_remain, test_size=val_frac/(1-test_frac), 
                                    stratify=train_remain['label'], random_state=seed+i)
        
        # 合并原始验证集到验证集
        val = pd.concat([val, valid_df], ignore_index=True)
        
        # 创建分割文件
        create_split_file(train, val, test, output_dir, i)
        
        # 输出统计信息
        print_split_stats(train, val, test, i)

def create_stratified_body_part_splits(df, output_dir, k=5, val_frac=0.1, test_frac=0.2, seed=42):
    """
    创建按身体部位分层的分割
    
    参数:
        df: 数据集DataFrame
        output_dir: 输出目录
        k: 交叉验证折数
        val_frac: 验证集比例
        test_frac: 测试集比例
        seed: 随机种子
    """
    # 保留原始MURA的训练/验证分割
    train_df = df[df['split'] == 'train'].copy()
    valid_df = df[df['split'] == 'valid'].copy()
    
    # 为每个折创建分割
    for i in range(k):
        # 设置随机种子
        np.random.seed(seed + i)
        
        # 按身体部位和标签进行分层抽样
        train_df['stratify'] = train_df.apply(lambda x: f"{x['body_part']}_{x['label']}", axis=1)
        
        # 从训练集中分出测试集
        train_remain, test = train_test_split(train_df, test_size=test_frac, 
                                            stratify=train_df['stratify'], random_state=seed+i)
        
        # 从剩余训练集中分出验证集
        train_remain['stratify'] = train_remain.apply(lambda x: f"{x['body_part']}_{x['label']}", axis=1)
        train, val = train_test_split(train_remain, test_size=val_frac/(1-test_frac), 
                                    stratify=train_remain['stratify'], random_state=seed+i)
        
        # 合并原始验证集到验证集
        val = pd.concat([val, valid_df], ignore_index=True)
        
        # 创建分割文件
        create_split_file(train, val, test, output_dir, i)
        
        # 输出统计信息
        print_split_stats(train, val, test, i)

def create_patient_level_splits(df, output_dir, k=5, val_frac=0.1, test_frac=0.2, seed=42):
    """
    创建患者级别的分割（确保同一患者的所有研究在同一分割中）
    
    参数:
        df: 数据集DataFrame
        output_dir: 输出目录
        k: 交叉验证折数
        val_frac: 验证集比例
        test_frac: 测试集比例
        seed: 随机种子
    """
    # 确保有患者ID
    df = extract_patient_id(df)
    
    # 保留原始MURA的训练/验证分割
    train_df = df[df['split'] == 'train'].copy()
    valid_df = df[df['split'] == 'valid'].copy()
    
    # 获取唯一患者ID
    train_patients = train_df['case_id'].unique()
    
    # 为每个折创建分割
    for i in range(k):
        # 设置随机种子
        np.random.seed(seed + i)
        
        # 创建患者级别的分层标签
        patient_labels = {}
        for patient in train_patients:
            # 获取该患者的所有研究
            patient_studies = train_df[train_df['case_id'] == patient]
            # 使用多数标签作为患者标签
            patient_labels[patient] = patient_studies['label'].mode()[0]
        
        # 创建患者DataFrame用于分割
        patient_df = pd.DataFrame({
            'case_id': list(patient_labels.keys()),
            'label': list(patient_labels.values())
        })
        
        # 分割患者
        patient_train_remain, patient_test = train_test_split(
            patient_df, test_size=test_frac, stratify=patient_df['label'], random_state=seed+i
        )
        
        patient_train, patient_val = train_test_split(
            patient_train_remain, test_size=val_frac/(1-test_frac), 
            stratify=patient_train_remain['label'], random_state=seed+i
        )
        
        # 根据患者分割获取研究分割
        train = train_df[train_df['case_id'].isin(patient_train['case_id'])]
        val = train_df[train_df['case_id'].isin(patient_val['case_id'])]
        test = train_df[train_df['case_id'].isin(patient_test['case_id'])]
        
        # 合并原始验证集到验证集
        val = pd.concat([val, valid_df], ignore_index=True)
        
        # 创建分割文件
        create_split_file(train, val, test, output_dir, i)
        
        # 输出统计信息
        print_split_stats(train, val, test, i)
        
        # 验证患者级别分割的正确性
        train_patients = set(train['case_id'])
        val_patients = set(val['case_id']) - set(valid_df['case_id'])  # 排除原始验证集
        test_patients = set(test['case_id'])
        
        # 检查患者重叠
        train_val_overlap = train_patients.intersection(val_patients)
        train_test_overlap = train_patients.intersection(test_patients)
        val_test_overlap = val_patients.intersection(test_patients)
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("警告: 患者级别分割存在重叠!")
            if train_val_overlap:
                print(f"训练集和验证集重叠: {len(train_val_overlap)} 个患者")
            if train_test_overlap:
                print(f"训练集和测试集重叠: {len(train_test_overlap)} 个患者")
            if val_test_overlap:
                print(f"验证集和测试集重叠: {len(val_test_overlap)} 个患者")

def create_balanced_splits(df, output_dir, k=5, val_frac=0.1, test_frac=0.2, seed=42):
    """
    创建平衡采样的分割（确保各身体部位和标签在各分割中的比例一致）
    
    参数:
        df: 数据集DataFrame
        output_dir: 输出目录
        k: 交叉验证折数
        val_frac: 验证集比例
        test_frac: 测试集比例
        seed: 随机种子
    """
    # 保留原始MURA的训练/验证分割
    train_df = df[df['split'] == 'train'].copy()
    valid_df = df[df['split'] == 'valid'].copy()
    
    # 为每个折创建分割
    for i in range(k):
        # 设置随机种子
        np.random.seed(seed + i)
        
        # 按身体部位分组
        body_parts = train_df['body_part'].unique()
        train_parts = []
        val_parts = []
        test_parts = []
        
        # 对每个身体部位单独进行分割
        for body_part in body_parts:
            part_df = train_df[train_df['body_part'] == body_part]
            
            # 按标签分层
            part_train_remain, part_test = train_test_split(
                part_df, test_size=test_frac, stratify=part_df['label'], random_state=seed+i
            )
            
            part_train, part_val = train_test_split(
                part_train_remain, test_size=val_frac/(1-test_frac), 
                stratify=part_train_remain['label'], random_state=seed+i
            )
            
            # 添加到相应列表
            train_parts.append(part_train)
            val_parts.append(part_val)
            test_parts.append(part_test)
        
        # 合并各部位的分割
        train = pd.concat(train_parts, ignore_index=True)
        val = pd.concat(val_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        
        # 合并原始验证集到验证集
        val = pd.concat([val, valid_df], ignore_index=True)
        
        # 创建分割文件
        create_split_file(train, val, test, output_dir, i)
        
        # 输出统计信息
        print_split_stats(train, val, test, i)

def create_focus_body_part_splits(df, output_dir, focus_body_part, k=5, val_frac=0.1, test_frac=0.2, seed=42):
    """
    创建专注于特定身体部位的分割
    
    参数:
        df: 数据集DataFrame
        output_dir: 输出目录
        focus_body_part: 重点关注的身体部位
        k: 交叉验证折数
        val_frac: 验证集比例
        test_frac: 测试集比例
        seed: 随机种子
    """
    # 过滤出指定身体部位的数据
    focus_df = df[df['body_part'] == focus_body_part].copy()
    
    if len(focus_df) == 0:
        print(f"错误: 未找到身体部位 '{focus_body_part}'")
        return
    
    # 保留原始MURA的训练/验证分割
    train_df = focus_df[focus_df['split'] == 'train'].copy()
    valid_df = focus_df[focus_df['split'] == 'valid'].copy()
    
    print(f"\n创建 {focus_body_part} 专用分割:")
    print(f"训练集: {len(train_df)} 样本")
    print(f"验证集: {len(valid_df)} 样本")
    print(f"总计: {len(focus_df)} 样本")
    
    # 为每个折创建分割
    for i in range(k):
        # 设置随机种子
        np.random.seed(seed + i)
        
        # 从训练集中分出测试集
        train_remain, test = train_test_split(train_df, test_size=test_frac, 
                                            stratify=train_df['label'], random_state=seed+i)
        
        # 从剩余训练集中分出验证集
        train, val = train_test_split(train_remain, test_size=val_frac/(1-test_frac), 
                                    stratify=train_remain['label'], random_state=seed+i)
        
        # 合并原始验证集到验证集
        val = pd.concat([val, valid_df], ignore_index=True)
        
        # 创建分割文件
        output_subdir = os.path.join(output_dir, focus_body_part)
        os.makedirs(output_subdir, exist_ok=True)
        create_split_file(train, val, test, output_subdir, i)
        
        # 输出统计信息
        print(f"\n{focus_body_part} 分割 {i}:")
        print(f"训练集: {len(train)} 样本 (正常: {sum(train['label'] == 0)}, 异常: {sum(train['label'] == 1)})")
        print(f"验证集: {len(val)} 样本 (正常: {sum(val['label'] == 0)}, 异常: {sum(val['label'] == 1)})")
        print(f"测试集: {len(test)} 样本 (正常: {sum(test['label'] == 0)}, 异常: {sum(test['label'] == 1)})")

def create_split_file(train, val, test, output_dir, fold_idx):
    """
    创建分割文件
    
    参数:
        train: 训练集DataFrame
        val: 验证集DataFrame
        test: 测试集DataFrame
        output_dir: 输出目录
        fold_idx: 折索引
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建分割文件
    splits = pd.DataFrame({
        'train': pd.Series(train['slide_id'].tolist() + [None] * (len(test) + len(val) - len(train))),
        'val': pd.Series(val['slide_id'].tolist() + [None] * (len(train) + len(test) - len(val))),
        'test': pd.Series(test['slide_id'].tolist() + [None] * (len(train) + len(val) - len(test)))
    })
    
    # 保存分割文件
    splits.to_csv(os.path.join(output_dir, f'splits_{fold_idx}.csv'), index=False)

def print_split_stats(train, val, test, fold_idx):
    """
    打印分割统计信息
    
    参数:
        train: 训练集DataFrame
        val: 验证集DataFrame
        test: 测试集DataFrame
        fold_idx: 折索引
    """
    print(f"\n分割 {fold_idx}:")
    print(f"训练集: {len(train)} 样本")
    print(f"验证集: {len(val)} 样本")
    print(f"测试集: {len(test)} 样本")
    
    # 按身体部位统计
    print("\n按身体部位分布:")
    for split_name, split_df in zip(['训练集', '验证集', '测试集'], [train, val, test]):
        print(f"\n{split_name}:")
        print(split_df.groupby(['body_part', 'label']).size().unstack())

def visualize_splits(df, output_dir, split_strategy, k=5):
    """
    可视化分割结果
    
    参数:
        df: 数据集DataFrame
        output_dir: 输出目录
        split_strategy: 分割策略
        k: 交叉验证折数
    """
    # 创建可视化输出目录
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 获取所有身体部位
    body_parts = df['body_part'].unique()
    
    # 为第一折创建可视化
    fold_idx = 0
    splits_file = os.path.join(output_dir, f'splits_{fold_idx}.csv')
    
    if not os.path.exists(splits_file):
        print(f"警告: 分割文件不存在 {splits_file}")
        return
    
    # 读取分割文件
    splits = pd.read_csv(splits_file)
    
    # 获取各分割的slide_id
    train_ids = splits['train'].dropna().tolist()
    val_ids = splits['val'].dropna().tolist()
    test_ids = splits['test'].dropna().tolist()
    
    # 创建分割标签
    df['split_label'] = 'unknown'
    df.loc[df['slide_id'].isin(train_ids), 'split_label'] = 'train'
    df.loc[df['slide_id'].isin(val_ids), 'split_label'] = 'val'
    df.loc[df['slide_id'].isin(test_ids), 'split_label'] = 'test'
    
    # 按身体部位和标签可视化分割
    plt.figure(figsize=(15, 10))
    
    # 创建计数表
    split_counts = df.groupby(['body_part', 'label', 'split_label']).size().unstack(fill_value=0)
    
    # 绘制堆叠条形图
    split_counts.plot(kind='bar', stacked=True)
    plt.title(f'分割策略: {split_strategy} - 按身体部位和标签的分布')
    plt.xlabel('身体部位和标签')
    plt.ylabel('样本数量')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{split_strategy}_body_part_label_dist.png'))
    
    # 按分割和标签可视化
    plt.figure(figsize=(10, 6))
    split_label_counts = df.groupby(['split_label', 'label']).size().unstack(fill_value=0)
    split_label_counts.plot(kind='bar')
    plt.title(f'分割策略: {split_strategy} - 按分割和标签的分布')
    plt.xlabel('分割')
    plt.ylabel('样本数量')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{split_strategy}_split_label_dist.png'))
    
    # 热力图展示各身体部位在各分割中的分布
    plt.figure(figsize=(12, 8))
    body_part_split = pd.crosstab(df['body_part'], df['split_label'])
    body_part_split_norm = body_part_split.div(body_part_split.sum(axis=1), axis=0)
    sns.heatmap(body_part_split_norm, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title(f'分割策略: {split_strategy} - 各身体部位在各分割中的比例')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{split_strategy}_body_part_split_heatmap.png'))
    
    print(f"\n可视化结果已保存到: {vis_dir}")

def validate_splits(output_dir, csv_path, k=5):
    """
    验证分割的有效性
    
    参数:
        output_dir: 分割文件目录
        csv_path: 原始CSV文件路径
        k: 交叉验证折数
    """
    # 读取原始CSV
    df = pd.read_csv(csv_path)
    
    # 检查每个折
    for i in range(k):
        splits_file = os.path.join(output_dir, f'splits_{i}.csv')
        
        if not os.path.exists(splits_file):
            print(f"警告: 分割文件不存在 {splits_file}")
            continue
        
        # 读取分割文件
        splits = pd.read_csv(splits_file)
        
        # 获取各分割的slide_id
        train_ids = set(splits['train'].dropna().tolist())
        val_ids = set(splits['val'].dropna().tolist())
        test_ids = set(splits['test'].dropna().tolist())
        
        # 检查重叠
        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)
        
        # 报告重叠情况
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"\n警告: 分割 {i} 存在重叠!")
            if train_val_overlap:
                print(f"训练集和验证集重叠: {len(train_val_overlap)} 个样本")
            if train_test_overlap:
                print(f"训练集和测试集重叠: {len(train_test_overlap)} 个样本")
            if val_test_overlap:
                print(f"验证集和测试集重叠: {len(val_test_overlap)} 个样本")
        else:
            print(f"\n分割 {i} 验证通过: 无重叠")
        
        # 检查所有样本是否都被包含
        all_split_ids = train_ids.union(val_ids).union(test_ids)
        all_df_ids = set(df['slide_id'].tolist())
        
        missing_ids = all_df_ids - all_split_ids
        extra_ids = all_split_ids - all_df_ids
        
        if missing_ids:
            print(f"警告: 分割 {i} 缺少 {len(missing_ids)} 个样本")
        
        if extra_ids:
            print(f"警告: 分割 {i} 包含 {len(extra_ids)} 个未知样本")

def create_splits(csv_path, output_dir, split_strategy='stratified_body_part', k=5, 
                val_frac=0.1, test_frac=0.2, seed=42, focus_body_part=None, visualize=False):
    """
    创建数据集分割
    
    参数:
        csv_path: MURA数据集CSV文件路径
        output_dir: 输出分割文件目录
        split_strategy: 分割策略
        k: 交叉验证折数
        val_frac: 验证集比例
        test_frac: 测试集比例
        seed: 随机种子
        focus_body_part: 重点关注的身体部位
        visualize: 是否可视化分割结果
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据集CSV
    df = pd.read_csv(csv_path)
    
    print(f"数据集总样本数: {len(df)}")
    print(f"训练集样本数: {len(df[df['split'] == 'train'])}")
    print(f"验证集样本数: {len(df[df['split'] == 'valid'])}")
    
    # 如果指定了重点关注的身体部位
    if focus_body_part:
        create_focus_body_part_splits(df, output_dir, focus_body_part, k, val_frac, test_frac, seed)
        return
    
    # 根据分割策略创建分割
    if split_strategy == 'simple':
        create_simple_splits(df, output_dir, k, val_frac, test_frac, seed)
    elif split_strategy == 'stratified_body_part':
        create_stratified_body_part_splits(df, output_dir, k, val_frac, test_frac, seed)
    elif split_strategy == 'patient_level':
        create_patient_level_splits(df, output_dir, k, val_frac, test_frac, seed)
    elif split_strategy == 'balanced':
        create_balanced_splits(df, output_dir, k, val_frac, test_frac, seed)
    else:
        print(f"错误: 未知的分割策略 '{split_strategy}'")
        return
    
    print(f"\n分割文件已保存到: {output_dir}")
    
    # 验证分割
    validate_splits(output_dir, csv_path, k)
    
    # 可视化分割结果
    if visualize:
        visualize_splits(df, output_dir, split_strategy, k)

def main():
    """主函数"""
    args = parse_args()
    create_splits(args.csv_path, args.output_dir, args.split_strategy, args.k, 
                args.val_frac, args.test_frac, args.seed, args.focus_body_part, args.visualize)
    print("MURA数据集分割创建完成!")

if __name__ == "__main__":
    main()