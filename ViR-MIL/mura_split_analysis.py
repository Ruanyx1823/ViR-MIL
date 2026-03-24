"""
MURA数据集分割分析工具
分析不同分割策略的统计特性和质量
"""

import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA数据集分割分析')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/mura_abnormality_detection.csv',
                        help='MURA数据集CSV文件路径')
    parser.add_argument('--splits_dirs', type=str, nargs='+', 
                        default=['splits/task_mura_abnormality_detection_100'],
                        help='分割文件目录列表')
    parser.add_argument('--split_names', type=str, nargs='+', default=None,
                        help='分割名称列表（用于图例）')
    parser.add_argument('--output_dir', type=str, default='results/split_analysis',
                        help='输出目录')
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='特征目录（用于特征分布分析）')
    return parser.parse_args()

def load_all_splits(csv_path, splits_dir, k=5):
    """
    加载所有分割
    
    参数:
        csv_path: MURA数据集CSV文件路径
        splits_dir: 分割文件目录
        k: 交叉验证折数
    
    返回:
        all_splits: 所有分割的字典
    """
    # 读取原始CSV
    df = pd.read_csv(csv_path)
    
    # 存储所有分割
    all_splits = {}
    
    # 加载每个折的分割
    for i in range(k):
        splits_file = os.path.join(splits_dir, f'splits_{i}.csv')
        if not os.path.exists(splits_file):
            print(f"警告: 分割文件不存在 {splits_file}")
            continue
        
        # 读取分割文件
        splits = pd.read_csv(splits_file)
        
        # 获取各分割的slide_id
        train_ids = set(splits['train'].dropna().tolist())
        val_ids = set(splits['val'].dropna().tolist())
        test_ids = set(splits['test'].dropna().tolist())
        
        # 存储分割
        all_splits[i] = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
    
    return all_splits, df

def analyze_split_overlap(all_splits):
    """
    分析分割重叠
    
    参数:
        all_splits: 所有分割的字典
    
    返回:
        overlap_stats: 重叠统计信息
    """
    overlap_stats = {
        'train_val': [],
        'train_test': [],
        'val_test': []
    }
    
    # 分析每个折的重叠
    for fold_idx, splits in all_splits.items():
        train_ids = splits['train']
        val_ids = splits['val']
        test_ids = splits['test']
        
        # 计算重叠
        train_val_overlap = len(train_ids.intersection(val_ids))
        train_test_overlap = len(train_ids.intersection(test_ids))
        val_test_overlap = len(val_ids.intersection(test_ids))
        
        # 存储重叠统计
        overlap_stats['train_val'].append(train_val_overlap)
        overlap_stats['train_test'].append(train_test_overlap)
        overlap_stats['val_test'].append(val_test_overlap)
    
    return overlap_stats

def analyze_label_balance(all_splits, df):
    """
    分析标签平衡性
    
    参数:
        all_splits: 所有分割的字典
        df: 原始DataFrame
    
    返回:
        label_stats: 标签统计信息
    """
    label_stats = defaultdict(list)
    
    # 分析每个折的标签平衡性
    for fold_idx, splits in all_splits.items():
        for split_name, split_ids in splits.items():
            # 获取该分割的数据
            split_df = df[df['slide_id'].isin(split_ids)]
            
            # 计算标签比例
            if len(split_df) > 0:
                normal_ratio = sum(split_df['label'] == 0) / len(split_df)
                abnormal_ratio = sum(split_df['label'] == 1) / len(split_df)
                
                # 存储标签统计
                label_stats[f'{split_name}_normal_ratio'].append(normal_ratio)
                label_stats[f'{split_name}_abnormal_ratio'].append(abnormal_ratio)
    
    return label_stats

def analyze_body_part_balance(all_splits, df):
    """
    分析身体部位平衡性
    
    参数:
        all_splits: 所有分割的字典
        df: 原始DataFrame
    
    返回:
        body_part_stats: 身体部位统计信息
    """
    body_part_stats = defaultdict(list)
    
    # 获取所有身体部位
    body_parts = df['body_part'].unique()
    
    # 分析每个折的身体部位平衡性
    for fold_idx, splits in all_splits.items():
        for split_name, split_ids in splits.items():
            # 获取该分割的数据
            split_df = df[df['slide_id'].isin(split_ids)]
            
            # 计算各身体部位比例
            if len(split_df) > 0:
                for body_part in body_parts:
                    body_part_ratio = sum(split_df['body_part'] == body_part) / len(split_df)
                    body_part_stats[f'{split_name}_{body_part}_ratio'].append(body_part_ratio)
    
    return body_part_stats

def analyze_patient_distribution(all_splits, df):
    """
    分析患者分布
    
    参数:
        all_splits: 所有分割的字典
        df: 原始DataFrame
    
    返回:
        patient_stats: 患者统计信息
    """
    patient_stats = defaultdict(list)
    
    # 确保有患者ID
    if 'case_id' not in df.columns:
        print("警告: DataFrame中没有case_id列，无法分析患者分布")
        return patient_stats
    
    # 分析每个折的患者分布
    for fold_idx, splits in all_splits.items():
        # 获取各分割的患者
        train_patients = set(df[df['slide_id'].isin(splits['train'])]['case_id'])
        val_patients = set(df[df['slide_id'].isin(splits['val'])]['case_id'])
        test_patients = set(df[df['slide_id'].isin(splits['test'])]['case_id'])
        
        # 计算患者重叠
        train_val_overlap = len(train_patients.intersection(val_patients))
        train_test_overlap = len(train_patients.intersection(test_patients))
        val_test_overlap = len(val_patients.intersection(test_patients))
        
        # 存储患者统计
        patient_stats['train_val_patient_overlap'].append(train_val_overlap)
        patient_stats['train_test_patient_overlap'].append(train_test_overlap)
        patient_stats['val_test_patient_overlap'].append(val_test_overlap)
        
        # 计算每个分割的患者数
        patient_stats['train_patients'].append(len(train_patients))
        patient_stats['val_patients'].append(len(val_patients))
        patient_stats['test_patients'].append(len(test_patients))
    
    return patient_stats

def analyze_feature_distribution(all_splits, df, feature_dir):
    """
    分析特征分布
    
    参数:
        all_splits: 所有分割的字典
        df: 原始DataFrame
        feature_dir: 特征目录
    
    返回:
        feature_stats: 特征统计信息
    """
    if not feature_dir or not os.path.exists(feature_dir):
        print(f"警告: 特征目录不存在 {feature_dir}")
        return {}
    
    import h5py
    
    feature_stats = defaultdict(list)
    
    # 分析每个折的特征分布
    for fold_idx, splits in all_splits.items():
        # 收集各分割的特征
        split_features = {}
        
        for split_name, split_ids in splits.items():
            # 获取该分割的数据
            split_df = df[df['slide_id'].isin(split_ids)]
            
            # 收集特征
            features = []
            
            # 限制样本数量以提高效率
            max_samples = 100
            sample_df = split_df.sample(min(len(split_df), max_samples), random_state=42)
            
            for idx, row in sample_df.iterrows():
                # 构建特征文件路径
                split = row['split']
                label_name = 'abnormal' if row['label'] == 1 else 'normal'
                body_part = row['body_part']
                slide_id = row['slide_id']
                
                # 尝试找到特征文件
                feature_path = os.path.join(feature_dir, split, label_name, f"{body_part}_{row['case_id']}_{slide_id}.h5")
                
                if not os.path.exists(feature_path):
                    # 尝试其他可能的路径格式
                    feature_path = os.path.join(feature_dir, f"{split}/{label_name}/{slide_id}.h5")
                    
                    if not os.path.exists(feature_path):
                        continue
                
                try:
                    # 读取特征
                    with h5py.File(feature_path, 'r') as f:
                        feature = f['features'][:]
                        # 使用平均特征
                        feature = np.mean(feature, axis=0)
                        features.append(feature)
                except Exception as e:
                    print(f"警告: 无法读取特征文件 {feature_path}: {e}")
            
            if features:
                # 存储特征
                split_features[split_name] = np.array(features)
        
        # 计算各分割间的特征距离
        for split1 in ['train', 'val', 'test']:
            for split2 in ['train', 'val', 'test']:
                if split1 != split2 and split1 in split_features and split2 in split_features:
                    # 计算平均特征
                    mean_feature1 = np.mean(split_features[split1], axis=0)
                    mean_feature2 = np.mean(split_features[split2], axis=0)
                    
                    # 计算欧氏距离
                    distance = np.linalg.norm(mean_feature1 - mean_feature2)
                    feature_stats[f'{split1}_{split2}_feature_distance'].append(distance)
    
    return feature_stats

def compare_splits(splits_dirs, split_names, csv_path, output_dir, feature_dir=None):
    """
    比较不同的分割策略
    
    参数:
        splits_dirs: 分割文件目录列表
        split_names: 分割名称列表
        csv_path: MURA数据集CSV文件路径
        output_dir: 输出目录
        feature_dir: 特征目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始CSV
    df = pd.read_csv(csv_path)
    
    # 如果没有提供分割名称，使用目录名
    if not split_names:
        split_names = [os.path.basename(d) for d in splits_dirs]
    
    # 确保分割名称和目录数量一致
    if len(split_names) != len(splits_dirs):
        print("警告: 分割名称数量与目录数量不一致，使用目录名")
        split_names = [os.path.basename(d) for d in splits_dirs]
    
    # 存储所有分割的统计信息
    all_stats = {}
    
    # 分析每个分割策略
    for i, (splits_dir, split_name) in enumerate(zip(splits_dirs, split_names)):
        print(f"分析分割策略: {split_name}")
        
        # 加载分割
        all_splits, _ = load_all_splits(csv_path, splits_dir)
        
        if not all_splits:
            print(f"警告: 未找到分割文件 {splits_dir}")
            continue
        
        # 分析分割重叠
        overlap_stats = analyze_split_overlap(all_splits)
        
        # 分析标签平衡性
        label_stats = analyze_label_balance(all_splits, df)
        
        # 分析身体部位平衡性
        body_part_stats = analyze_body_part_balance(all_splits, df)
        
        # 分析患者分布
        patient_stats = analyze_patient_distribution(all_splits, df)
        
        # 分析特征分布
        feature_stats = {}
        if feature_dir:
            feature_stats = analyze_feature_distribution(all_splits, df, feature_dir)
        
        # 合并所有统计信息
        stats = {}
        stats.update(overlap_stats)
        stats.update(label_stats)
        stats.update(body_part_stats)
        stats.update(patient_stats)
        stats.update(feature_stats)
        
        # 存储统计信息
        all_stats[split_name] = stats
    
    # 创建比较报告
    create_comparison_report(all_stats, output_dir)
    
    # 可视化比较结果
    visualize_comparison(all_stats, output_dir)

def create_comparison_report(all_stats, output_dir):
    """
    创建比较报告
    
    参数:
        all_stats: 所有分割的统计信息
        output_dir: 输出目录
    """
    # 创建报告文件
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("MURA数据集分割比较报告\n")
        f.write("=======================\n\n")
        
        # 比较分割重叠
        f.write("分割重叠比较:\n")
        for split_name, stats in all_stats.items():
            if 'train_val' in stats:
                f.write(f"  {split_name}:\n")
                f.write(f"    训练集-验证集重叠: {np.mean(stats['train_val']):.2f} ± {np.std(stats['train_val']):.2f}\n")
                f.write(f"    训练集-测试集重叠: {np.mean(stats['train_test']):.2f} ± {np.std(stats['train_test']):.2f}\n")
                f.write(f"    验证集-测试集重叠: {np.mean(stats['val_test']):.2f} ± {np.std(stats['val_test']):.2f}\n")
        f.write("\n")
        
        # 比较标签平衡性
        f.write("标签平衡性比较:\n")
        for split_name, stats in all_stats.items():
            if 'train_normal_ratio' in stats:
                f.write(f"  {split_name}:\n")
                f.write(f"    训练集正常比例: {np.mean(stats['train_normal_ratio']):.2f} ± {np.std(stats['train_normal_ratio']):.2f}\n")
                f.write(f"    训练集异常比例: {np.mean(stats['train_abnormal_ratio']):.2f} ± {np.std(stats['train_abnormal_ratio']):.2f}\n")
                f.write(f"    验证集正常比例: {np.mean(stats['val_normal_ratio']):.2f} ± {np.std(stats['val_normal_ratio']):.2f}\n")
                f.write(f"    验证集异常比例: {np.mean(stats['val_abnormal_ratio']):.2f} ± {np.std(stats['val_abnormal_ratio']):.2f}\n")
                f.write(f"    测试集正常比例: {np.mean(stats['test_normal_ratio']):.2f} ± {np.std(stats['test_normal_ratio']):.2f}\n")
                f.write(f"    测试集异常比例: {np.mean(stats['test_abnormal_ratio']):.2f} ± {np.std(stats['test_abnormal_ratio']):.2f}\n")
        f.write("\n")
        
        # 比较患者分布
        f.write("患者分布比较:\n")
        for split_name, stats in all_stats.items():
            if 'train_val_patient_overlap' in stats:
                f.write(f"  {split_name}:\n")
                f.write(f"    训练集-验证集患者重叠: {np.mean(stats['train_val_patient_overlap']):.2f} ± {np.std(stats['train_val_patient_overlap']):.2f}\n")
                f.write(f"    训练集-测试集患者重叠: {np.mean(stats['train_test_patient_overlap']):.2f} ± {np.std(stats['train_test_patient_overlap']):.2f}\n")
                f.write(f"    验证集-测试集患者重叠: {np.mean(stats['val_test_patient_overlap']):.2f} ± {np.std(stats['val_test_patient_overlap']):.2f}\n")
        f.write("\n")
        
        # 比较特征分布
        f.write("特征分布比较:\n")
        for split_name, stats in all_stats.items():
            if 'train_val_feature_distance' in stats:
                f.write(f"  {split_name}:\n")
                f.write(f"    训练集-验证集特征距离: {np.mean(stats['train_val_feature_distance']):.4f} ± {np.std(stats['train_val_feature_distance']):.4f}\n")
                f.write(f"    训练集-测试集特征距离: {np.mean(stats['train_test_feature_distance']):.4f} ± {np.std(stats['train_test_feature_distance']):.4f}\n")
                f.write(f"    验证集-测试集特征距离: {np.mean(stats['val_test_feature_distance']):.4f} ± {np.std(stats['val_test_feature_distance']):.4f}\n")
    
    print(f"比较报告已保存到: {report_path}")

def visualize_comparison(all_stats, output_dir):
    """
    可视化比较结果
    
    参数:
        all_stats: 所有分割的统计信息
        output_dir: 输出目录
    """
    # 绘制标签平衡性比较
    plt.figure(figsize=(12, 8))
    
    split_names = list(all_stats.keys())
    x = np.arange(len(split_names))
    width = 0.15
    
    # 绘制训练集标签比例
    train_normal = [np.mean(all_stats[name]['train_normal_ratio']) if 'train_normal_ratio' in all_stats[name] else 0 for name in split_names]
    train_abnormal = [np.mean(all_stats[name]['train_abnormal_ratio']) if 'train_abnormal_ratio' in all_stats[name] else 0 for name in split_names]
    
    # 绘制验证集标签比例
    val_normal = [np.mean(all_stats[name]['val_normal_ratio']) if 'val_normal_ratio' in all_stats[name] else 0 for name in split_names]
    val_abnormal = [np.mean(all_stats[name]['val_abnormal_ratio']) if 'val_abnormal_ratio' in all_stats[name] else 0 for name in split_names]
    
    # 绘制测试集标签比例
    test_normal = [np.mean(all_stats[name]['test_normal_ratio']) if 'test_normal_ratio' in all_stats[name] else 0 for name in split_names]
    test_abnormal = [np.mean(all_stats[name]['test_abnormal_ratio']) if 'test_abnormal_ratio' in all_stats[name] else 0 for name in split_names]
    
    plt.bar(x - 2.5*width, train_normal, width, label='训练集正常')
    plt.bar(x - 1.5*width, train_abnormal, width, label='训练集异常')
    plt.bar(x - 0.5*width, val_normal, width, label='验证集正常')
    plt.bar(x + 0.5*width, val_abnormal, width, label='验证集异常')
    plt.bar(x + 1.5*width, test_normal, width, label='测试集正常')
    plt.bar(x + 2.5*width, test_abnormal, width, label='测试集异常')
    
    plt.xlabel('分割策略')
    plt.ylabel('标签比例')
    plt.title('不同分割策略的标签平衡性比较')
    plt.xticks(x, split_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_balance_comparison.png'))
    
    # 绘制患者重叠比较
    plt.figure(figsize=(10, 6))
    
    # 绘制患者重叠
    train_val_overlap = [np.mean(all_stats[name]['train_val_patient_overlap']) if 'train_val_patient_overlap' in all_stats[name] else 0 for name in split_names]
    train_test_overlap = [np.mean(all_stats[name]['train_test_patient_overlap']) if 'train_test_patient_overlap' in all_stats[name] else 0 for name in split_names]
    val_test_overlap = [np.mean(all_stats[name]['val_test_patient_overlap']) if 'val_test_patient_overlap' in all_stats[name] else 0 for name in split_names]
    
    plt.bar(x - width, train_val_overlap, width, label='训练集-验证集')
    plt.bar(x, train_test_overlap, width, label='训练集-测试集')
    plt.bar(x + width, val_test_overlap, width, label='验证集-测试集')
    
    plt.xlabel('分割策略')
    plt.ylabel('患者重叠数量')
    plt.title('不同分割策略的患者重叠比较')
    plt.xticks(x, split_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_overlap_comparison.png'))
    
    # 绘制特征距离比较
    if 'train_val_feature_distance' in all_stats[split_names[0]]:
        plt.figure(figsize=(10, 6))
        
        # 绘制特征距离
        train_val_dist = [np.mean(all_stats[name]['train_val_feature_distance']) if 'train_val_feature_distance' in all_stats[name] else 0 for name in split_names]
        train_test_dist = [np.mean(all_stats[name]['train_test_feature_distance']) if 'train_test_feature_distance' in all_stats[name] else 0 for name in split_names]
        val_test_dist = [np.mean(all_stats[name]['val_test_feature_distance']) if 'val_test_feature_distance' in all_stats[name] else 0 for name in split_names]
        
        plt.bar(x - width, train_val_dist, width, label='训练集-验证集')
        plt.bar(x, train_test_dist, width, label='训练集-测试集')
        plt.bar(x + width, val_test_dist, width, label='验证集-测试集')
        
        plt.xlabel('分割策略')
        plt.ylabel('特征距离')
        plt.title('不同分割策略的特征距离比较')
        plt.xticks(x, split_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distance_comparison.png'))

def main():
    """主函数"""
    args = parse_args()
    compare_splits(args.splits_dirs, args.split_names, args.csv_path, args.output_dir, args.feature_dir)
    print("MURA数据集分割分析完成!")

if __name__ == "__main__":
    main()