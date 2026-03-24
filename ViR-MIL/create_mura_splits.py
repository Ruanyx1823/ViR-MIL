"""
为MURA数据集创建训练/验证/测试分割
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='创建MURA数据集分割')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/mura_abnormality_detection.csv',
                        help='MURA数据集CSV文件路径')
    parser.add_argument('--split_dir', '--output_dir', type=str, default='splits/task_mura_abnormality_detection_100',
                        help='输出分割文件目录')
    parser.add_argument('--k', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_frac', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--stratify_by_body_part', action='store_true',
                        help='是否按身体部位进行分层抽样')
    return parser.parse_args()

def create_splits(csv_path, output_dir, k=5, val_frac=0.1, test_frac=0.2, seed=42, stratify_by_body_part=False):
    """
    创建数据集分割
    
    参数:
        csv_path: MURA数据集CSV文件路径
        output_dir: 输出分割文件目录
        k: 交叉验证折数
        val_frac: 验证集比例
        test_frac: 测试集比例
        seed: 随机种子
        stratify_by_body_part: 是否按身体部位进行分层抽样
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据集CSV
    df = pd.read_csv(csv_path)
    
    # 保留原始MURA的训练/验证分割
    train_df = df[df['split'] == 'train'].copy()
    valid_df = df[df['split'] == 'valid'].copy()
    
    # 为每个折创建分割
    for i in range(k):
        # 设置随机种子
        np.random.seed(seed + i)
        
        # 从训练集中分出测试集
        if stratify_by_body_part:
            # 按身体部位和标签进行分层抽样
            stratify_cols = ['body_part', 'label']
            train_df['stratify'] = train_df.apply(lambda x: f"{x['body_part']}_{x['label']}", axis=1)
            train_remain, test = train_test_split(train_df, test_size=test_frac, 
                                                 stratify=train_df['stratify'], random_state=seed+i)
            
            # 从剩余训练集中分出验证集
            train_remain['stratify'] = train_remain.apply(lambda x: f"{x['body_part']}_{x['label']}", axis=1)
            train, val = train_test_split(train_remain, test_size=val_frac/(1-test_frac), 
                                         stratify=train_remain['stratify'], random_state=seed+i)
        else:
            # 仅按标签进行分层抽样
            train_remain, test = train_test_split(train_df, test_size=test_frac, 
                                                 stratify=train_df['label'], random_state=seed+i)
            train, val = train_test_split(train_remain, test_size=val_frac/(1-test_frac), 
                                         stratify=train_remain['label'], random_state=seed+i)
        
        # 合并原始验证集到验证集
        val = pd.concat([val, valid_df], ignore_index=True)
        
        # 创建分割文件
        splits = pd.DataFrame({
            'train': pd.Series(train['slide_id'].tolist() + [None] * (len(test) + len(val) - len(train))),
            'val': pd.Series(val['slide_id'].tolist() + [None] * (len(train) + len(test) - len(val))),
            'test': pd.Series(test['slide_id'].tolist() + [None] * (len(train) + len(val) - len(test)))
        })
        
        # 保存分割文件
        splits.to_csv(os.path.join(output_dir, f'splits_{i}.csv'), index=False)
        
        # 输出统计信息
        print(f"\n分割 {i}:")
        print(f"训练集: {len(train)} 样本")
        print(f"验证集: {len(val)} 样本")
        print(f"测试集: {len(test)} 样本")
        
        # 按身体部位统计
        print("\n按身体部位分布:")
        for split_name, split_df in zip(['训练集', '验证集', '测试集'], [train, val, test]):
            print(f"\n{split_name}:")
            print(split_df.groupby(['body_part', 'label']).size().unstack())
    
    print(f"\n分割文件已保存到: {output_dir}")

def main():
    """主函数"""
    args = parse_args()
    create_splits(args.csv_path, args.split_dir, args.k, args.val_frac, 
                 args.test_frac, args.seed, args.stratify_by_body_part)
    print("MURA数据集分割创建完成!")

if __name__ == "__main__":
    main()