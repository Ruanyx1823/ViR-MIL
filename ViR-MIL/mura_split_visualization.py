"""
MURA数据集分割可视化工具
用于可视化和分析数据分割的质量和特性
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA数据集分割可视化')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/mura_abnormality_detection.csv',
                        help='MURA数据集CSV文件路径')
    parser.add_argument('--splits_dir', type=str, default='splits/task_mura_abnormality_detection_100',
                        help='分割文件目录')
    parser.add_argument('--output_dir', type=str, default='results/split_visualization',
                        help='输出目录')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='要可视化的折索引')
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='特征目录（用于特征空间可视化）')
    return parser.parse_args()

def load_split_data(csv_path, splits_dir, fold_idx):
    """
    加载分割数据
    
    参数:
        csv_path: MURA数据集CSV文件路径
        splits_dir: 分割文件目录
        fold_idx: 折索引
    
    返回:
        df: 带有分割标签的DataFrame
    """
    # 读取原始CSV
    df = pd.read_csv(csv_path)
    
    # 读取分割文件
    splits_file = os.path.join(splits_dir, f'splits_{fold_idx}.csv')
    if not os.path.exists(splits_file):
        raise ValueError(f"分割文件不存在: {splits_file}")
    
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
    
    return df

def visualize_body_part_distribution(df, output_dir):
    """
    可视化身体部位分布
    
    参数:
        df: 带有分割标签的DataFrame
        output_dir: 输出目录
    """
    # 按身体部位和分割统计
    body_part_split = pd.crosstab(df['body_part'], df['split_label'])
    
    # 绘制堆叠条形图
    plt.figure(figsize=(12, 8))
    body_part_split.plot(kind='bar', stacked=True)
    plt.title('各身体部位在各分割中的分布')
    plt.xlabel('身体部位')
    plt.ylabel('样本数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'body_part_distribution.png'))
    
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    body_part_split_norm = body_part_split.div(body_part_split.sum(axis=1), axis=0)
    sns.heatmap(body_part_split_norm, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('各身体部位在各分割中的比例')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'body_part_distribution_heatmap.png'))

def visualize_label_distribution(df, output_dir):
    """
    可视化标签分布
    
    参数:
        df: 带有分割标签的DataFrame
        output_dir: 输出目录
    """
    # 按标签和分割统计
    label_split = pd.crosstab(df['label'], df['split_label'])
    
    # 绘制堆叠条形图
    plt.figure(figsize=(10, 6))
    label_split.plot(kind='bar')
    plt.title('各标签在各分割中的分布')
    plt.xlabel('标签')
    plt.ylabel('样本数量')
    plt.xticks([0, 1], ['正常', '异常'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
    
    # 绘制饼图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, split in enumerate(['train', 'val', 'test']):
        split_df = df[df['split_label'] == split]
        label_counts = split_df['label'].value_counts()
        axes[i].pie(label_counts, labels=['正常', '异常'], autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'{split} 集标签分布')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution_pie.png'))

def visualize_body_part_label_distribution(df, output_dir):
    """
    可视化身体部位和标签的联合分布
    
    参数:
        df: 带有分割标签的DataFrame
        output_dir: 输出目录
    """
    # 创建身体部位和标签的组合
    df['body_part_label'] = df.apply(lambda x: f"{x['body_part']}_{x['label']}", axis=1)
    
    # 按身体部位、标签和分割统计
    body_part_label_split = pd.crosstab(df['body_part_label'], df['split_label'])
    
    # 绘制堆叠条形图
    plt.figure(figsize=(15, 10))
    body_part_label_split.plot(kind='bar', stacked=True)
    plt.title('各身体部位和标签在各分割中的分布')
    plt.xlabel('身体部位和标签')
    plt.ylabel('样本数量')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'body_part_label_distribution.png'))
    
    # 为每个身体部位创建单独的图表
    body_parts = df['body_part'].unique()
    
    for body_part in body_parts:
        part_df = df[df['body_part'] == body_part]
        
        # 按标签和分割统计
        label_split = pd.crosstab(part_df['label'], part_df['split_label'])
        
        # 绘制条形图
        plt.figure(figsize=(10, 6))
        label_split.plot(kind='bar')
        plt.title(f'{body_part} 各标签在各分割中的分布')
        plt.xlabel('标签')
        plt.ylabel('样本数量')
        plt.xticks([0, 1], ['正常', '异常'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{body_part}_label_distribution.png'))

def visualize_patient_distribution(df, output_dir):
    """
    可视化患者分布
    
    参数:
        df: 带有分割标签的DataFrame
        output_dir: 输出目录
    """
    # 确保有患者ID
    if 'case_id' not in df.columns:
        print("警告: DataFrame中没有case_id列，无法可视化患者分布")
        return
    
    # 计算每个患者的研究数量
    patient_counts = df.groupby('case_id').size()
    
    # 绘制患者研究数量分布
    plt.figure(figsize=(10, 6))
    patient_counts.value_counts().sort_index().plot(kind='bar')
    plt.title('患者研究数量分布')
    plt.xlabel('每个患者的研究数量')
    plt.ylabel('患者数量')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_study_count_distribution.png'))
    
    # 检查患者在分割间的重叠
    patient_splits = {}
    for split in ['train', 'val', 'test']:
        split_df = df[df['split_label'] == split]
        patient_splits[split] = set(split_df['case_id'])
    
    # 计算重叠
    train_val_overlap = patient_splits['train'].intersection(patient_splits['val'])
    train_test_overlap = patient_splits['train'].intersection(patient_splits['test'])
    val_test_overlap = patient_splits['val'].intersection(patient_splits['test'])
    
    # 创建文本报告
    report = [
        "患者分布报告:",
        f"训练集患者数: {len(patient_splits['train'])}",
        f"验证集患者数: {len(patient_splits['val'])}",
        f"测试集患者数: {len(patient_splits['test'])}",
        f"训练集和验证集重叠患者数: {len(train_val_overlap)}",
        f"训练集和测试集重叠患者数: {len(train_test_overlap)}",
        f"验证集和测试集重叠患者数: {len(val_test_overlap)}"
    ]
    
    # 保存报告
    with open(os.path.join(output_dir, 'patient_distribution_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    # 创建韦恩图
    from matplotlib_venn import venn3
    plt.figure(figsize=(10, 10))
    venn3([patient_splits['train'], patient_splits['val'], patient_splits['test']], 
          ('训练集', '验证集', '测试集'))
    plt.title('各分割中的患者重叠情况')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_overlap_venn.png'))

def visualize_feature_space(df, feature_dir, output_dir):
    """
    可视化特征空间分布
    
    参数:
        df: 带有分割标签的DataFrame
        feature_dir: 特征目录
        output_dir: 输出目录
    """
    if not feature_dir or not os.path.exists(feature_dir):
        print(f"警告: 特征目录不存在 {feature_dir}")
        return
    
    import h5py
    
    # 收集特征
    features = []
    labels = []
    split_labels = []
    body_parts = []
    
    # 限制样本数量以避免过度拥挤
    max_samples = 500
    sample_df = df.sample(min(len(df), max_samples), random_state=42)
    
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
                labels.append(row['label'])
                split_labels.append(row['split_label'])
                body_parts.append(row['body_part'])
        except Exception as e:
            print(f"警告: 无法读取特征文件 {feature_path}: {e}")
    
    if not features:
        print("警告: 未找到任何特征文件")
        return
    
    # 转换为numpy数组
    features = np.array(features)
    labels = np.array(labels)
    split_labels = np.array(split_labels)
    body_parts = np.array(body_parts)
    
    # 使用PCA降维
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # 绘制PCA结果 - 按标签着色
    plt.figure(figsize=(12, 10))
    for label in np.unique(labels):
        label_name = '异常' if label == 1 else '正常'
        mask = labels == label
        plt.scatter(features_pca[mask, 0], features_pca[mask, 1], label=label_name, alpha=0.7)
    plt.title('特征空间PCA可视化 (按标签着色)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_space_pca_by_label.png'))
    
    # 绘制PCA结果 - 按分割着色
    plt.figure(figsize=(12, 10))
    for split in np.unique(split_labels):
        mask = split_labels == split
        plt.scatter(features_pca[mask, 0], features_pca[mask, 1], label=split, alpha=0.7)
    plt.title('特征空间PCA可视化 (按分割着色)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_space_pca_by_split.png'))
    
    # 绘制PCA结果 - 按身体部位着色
    plt.figure(figsize=(12, 10))
    for body_part in np.unique(body_parts):
        mask = body_parts == body_part
        plt.scatter(features_pca[mask, 0], features_pca[mask, 1], label=body_part, alpha=0.7)
    plt.title('特征空间PCA可视化 (按身体部位着色)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_space_pca_by_body_part.png'))
    
    # 使用t-SNE降维
    try:
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)
        
        # 绘制t-SNE结果 - 按标签着色
        plt.figure(figsize=(12, 10))
        for label in np.unique(labels):
            label_name = '异常' if label == 1 else '正常'
            mask = labels == label
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=label_name, alpha=0.7)
        plt.title('特征空间t-SNE可视化 (按标签着色)')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_space_tsne_by_label.png'))
        
        # 绘制t-SNE结果 - 按分割着色
        plt.figure(figsize=(12, 10))
        for split in np.unique(split_labels):
            mask = split_labels == split
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=split, alpha=0.7)
        plt.title('特征空间t-SNE可视化 (按分割着色)')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_space_tsne_by_split.png'))
        
        # 绘制t-SNE结果 - 按身体部位着色
        plt.figure(figsize=(12, 10))
        for body_part in np.unique(body_parts):
            mask = body_parts == body_part
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=body_part, alpha=0.7)
        plt.title('特征空间t-SNE可视化 (按身体部位着色)')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_space_tsne_by_body_part.png'))
    except Exception as e:
        print(f"警告: t-SNE可视化失败: {e}")

def main():
    """主函数"""
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载分割数据
    df = load_split_data(args.csv_path, args.splits_dir, args.fold_idx)
    
    print(f"可视化第 {args.fold_idx} 折的分割...")
    
    # 可视化身体部位分布
    visualize_body_part_distribution(df, args.output_dir)
    
    # 可视化标签分布
    visualize_label_distribution(df, args.output_dir)
    
    # 可视化身体部位和标签的联合分布
    visualize_body_part_label_distribution(df, args.output_dir)
    
    # 可视化患者分布
    visualize_patient_distribution(df, args.output_dir)
    
    # 如果提供了特征目录，可视化特征空间
    if args.feature_dir:
        visualize_feature_space(df, args.feature_dir, args.output_dir)
    
    print(f"可视化结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()