"""
MURA特征验证模块
验证提取的特征文件是否正确
"""

import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA特征验证')
    parser.add_argument('--high_res_dir', type=str, default='processed_data/high_res_features',
                        help='高分辨率特征目录')
    parser.add_argument('--low_res_dir', type=str, default='processed_data/low_res_features',
                        help='低分辨率特征目录')
    parser.add_argument('--output_dir', type=str, default='results/feature_verification',
                        help='输出目录')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/mura_abnormality_detection.csv',
                        help='数据集CSV文件路径')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化特征')
    return parser.parse_args()

def collect_feature_files(feature_dir):
    """
    收集特征文件
    
    参数:
        feature_dir: 特征目录
    
    返回:
        feature_files: 特征文件列表
    """
    feature_files = []
    
    for root, dirs, files in os.walk(feature_dir):
        for file in files:
            if file.endswith('.h5'):
                feature_files.append(os.path.join(root, file))
    
    return feature_files

def verify_feature_dimensions(feature_files):
    """
    验证特征维度
    
    参数:
        feature_files: 特征文件列表
    
    返回:
        dimensions: 维度列表
        shapes: 形状列表
    """
    dimensions = []
    shapes = []
    
    for file in tqdm(feature_files, desc="验证特征维度"):
        try:
            with h5py.File(file, 'r') as f:
                features = f['features'][:]
                dimensions.append(features.shape[1])
                shapes.append(features.shape)
        except Exception as e:
            print(f"警告: 无法读取文件 {file}: {e}")
    
    return dimensions, shapes

def verify_feature_consistency(high_res_files, low_res_files):
    """
    验证高低分辨率特征一致性
    
    参数:
        high_res_files: 高分辨率特征文件列表
        low_res_files: 低分辨率特征文件列表
    
    返回:
        is_consistent: 是否一致
        missing_high: 高分辨率缺少的文件
        missing_low: 低分辨率缺少的文件
    """
    # 提取文件名
    high_res_names = set([os.path.basename(f) for f in high_res_files])
    low_res_names = set([os.path.basename(f) for f in low_res_files])
    
    # 检查一致性
    is_consistent = (high_res_names == low_res_names)
    missing_high = low_res_names - high_res_names
    missing_low = high_res_names - low_res_names
    
    return is_consistent, missing_high, missing_low

def verify_feature_values(feature_files, sample_count=10):
    """
    验证特征值
    
    参数:
        feature_files: 特征文件列表
        sample_count: 样本数量
    
    返回:
        stats: 统计信息
    """
    stats = {
        'min_values': [],
        'max_values': [],
        'mean_values': [],
        'std_values': []
    }
    
    # 随机选择样本
    if len(feature_files) > sample_count:
        np.random.seed(42)
        sample_files = np.random.choice(feature_files, sample_count, replace=False)
    else:
        sample_files = feature_files
    
    for file in tqdm(sample_files, desc="验证特征值"):
        try:
            with h5py.File(file, 'r') as f:
                features = f['features'][:]
                stats['min_values'].append(np.min(features))
                stats['max_values'].append(np.max(features))
                stats['mean_values'].append(np.mean(features))
                stats['std_values'].append(np.std(features))
        except Exception as e:
            print(f"警告: 无法读取文件 {file}: {e}")
    
    return stats

def match_features_to_csv(feature_files, csv_path):
    """
    将特征文件与CSV文件匹配
    
    参数:
        feature_files: 特征文件列表
        csv_path: CSV文件路径
    
    返回:
        matched_count: 匹配数量
        total_count: 总数量
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 提取研究ID
    study_ids = set()
    for _, row in df.iterrows():
        body_part = row['body_part']
        case_id = row['case_id']
        slide_id = row['slide_id']
        study_id = f"{body_part}_{case_id}_{slide_id}"
        study_ids.add(study_id)
    
    # 匹配特征文件
    matched_count = 0
    for file in feature_files:
        basename = os.path.basename(file).replace('.h5', '')
        if basename in study_ids:
            matched_count += 1
    
    return matched_count, len(study_ids)

def visualize_features(high_res_files, low_res_files, output_dir, csv_path, sample_count=100):
    """
    可视化特征
    
    参数:
        high_res_files: 高分辨率特征文件列表
        low_res_files: 低分辨率特征文件列表
        output_dir: 输出目录
        csv_path: CSV文件路径
        sample_count: 样本数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建文件名到标签的映射
    filename_to_label = {}
    for _, row in df.iterrows():
        body_part = row['body_part']
        case_id = row['case_id']
        slide_id = row['slide_id']
        label = row['label']
        study_id = f"{body_part}_{case_id}_{slide_id}"
        filename_to_label[study_id] = label
    
    # 随机选择样本
    if len(high_res_files) > sample_count:
        np.random.seed(42)
        indices = np.random.choice(len(high_res_files), sample_count, replace=False)
        high_res_samples = [high_res_files[i] for i in indices]
        low_res_samples = [low_res_files[i] for i in indices]
    else:
        high_res_samples = high_res_files
        low_res_samples = low_res_files
    
    # 提取特征和标签
    high_res_features = []
    low_res_features = []
    labels = []
    
    for high_file, low_file in tqdm(zip(high_res_samples, low_res_samples), desc="提取特征"):
        try:
            # 提取文件名
            basename = os.path.basename(high_file).replace('.h5', '')
            
            # 获取标签
            if basename in filename_to_label:
                label = filename_to_label[basename]
            else:
                continue
            
            # 读取特征
            with h5py.File(high_file, 'r') as f:
                high_features = f['features'][:]
                high_res_features.append(np.mean(high_features, axis=0))
            
            with h5py.File(low_file, 'r') as f:
                low_features = f['features'][:]
                low_res_features.append(np.mean(low_features, axis=0))
            
            labels.append(label)
        except Exception as e:
            print(f"警告: 无法读取文件: {e}")
    
    # 转换为numpy数组
    high_res_features = np.array(high_res_features)
    low_res_features = np.array(low_res_features)
    labels = np.array(labels)
    
    # 使用PCA降维
    pca = PCA(n_components=2)
    high_res_pca = pca.fit_transform(high_res_features)
    low_res_pca = pca.fit_transform(low_res_features)
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    high_res_tsne = tsne.fit_transform(high_res_features)
    low_res_tsne = tsne.fit_transform(low_res_features)
    
    # 绘制PCA结果
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(high_res_pca[mask, 0], high_res_pca[mask, 1], label=f"Label {label}")
    plt.title("High Resolution Features (PCA)")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(low_res_pca[mask, 0], low_res_pca[mask, 1], label=f"Label {label}")
    plt.title("Low Resolution Features (PCA)")
    plt.legend()
    
    # 绘制t-SNE结果
    plt.subplot(2, 2, 3)
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(high_res_tsne[mask, 0], high_res_tsne[mask, 1], label=f"Label {label}")
    plt.title("High Resolution Features (t-SNE)")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(low_res_tsne[mask, 0], low_res_tsne[mask, 1], label=f"Label {label}")
    plt.title("Low Resolution Features (t-SNE)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_visualization.png"))
    plt.close()
    
    print(f"特征可视化已保存到 {os.path.join(output_dir, 'feature_visualization.png')}")

def main():
    """主函数"""
    args = parse_args()
    
    # 收集特征文件
    print("收集特征文件...")
    high_res_files = collect_feature_files(args.high_res_dir)
    low_res_files = collect_feature_files(args.low_res_dir)
    
    print(f"高分辨率特征文件: {len(high_res_files)}")
    print(f"低分辨率特征文件: {len(low_res_files)}")
    
    # 验证特征维度
    print("\n验证特征维度...")
    high_res_dims, high_res_shapes = verify_feature_dimensions(high_res_files)
    low_res_dims, low_res_shapes = verify_feature_dimensions(low_res_files)
    
    print(f"高分辨率特征维度: {np.unique(high_res_dims)}")
    print(f"低分辨率特征维度: {np.unique(low_res_dims)}")
    
    # 验证特征一致性
    print("\n验证特征一致性...")
    is_consistent, missing_high, missing_low = verify_feature_consistency(high_res_files, low_res_files)
    
    if is_consistent:
        print("高低分辨率特征文件完全匹配")
    else:
        print("警告: 高低分辨率特征文件不匹配")
        if missing_high:
            print(f"高分辨率缺少 {len(missing_high)} 个文件")
        if missing_low:
            print(f"低分辨率缺少 {len(missing_low)} 个文件")
    
    # 验证特征值
    print("\n验证特征值...")
    high_res_stats = verify_feature_values(high_res_files)
    low_res_stats = verify_feature_values(low_res_files)
    
    print("高分辨率特征统计:")
    print(f"最小值范围: {np.min(high_res_stats['min_values'])} - {np.max(high_res_stats['min_values'])}")
    print(f"最大值范围: {np.min(high_res_stats['max_values'])} - {np.max(high_res_stats['max_values'])}")
    print(f"均值范围: {np.min(high_res_stats['mean_values'])} - {np.max(high_res_stats['mean_values'])}")
    print(f"标准差范围: {np.min(high_res_stats['std_values'])} - {np.max(high_res_stats['std_values'])}")
    
    print("\n低分辨率特征统计:")
    print(f"最小值范围: {np.min(low_res_stats['min_values'])} - {np.max(low_res_stats['min_values'])}")
    print(f"最大值范围: {np.min(low_res_stats['max_values'])} - {np.max(low_res_stats['max_values'])}")
    print(f"均值范围: {np.min(low_res_stats['mean_values'])} - {np.max(low_res_stats['mean_values'])}")
    print(f"标准差范围: {np.min(low_res_stats['std_values'])} - {np.max(low_res_stats['std_values'])}")
    
    # 将特征文件与CSV文件匹配
    print("\n将特征文件与CSV文件匹配...")
    high_res_matched, total_count = match_features_to_csv(high_res_files, args.csv_path)
    low_res_matched, _ = match_features_to_csv(low_res_files, args.csv_path)
    
    print(f"高分辨率特征文件匹配: {high_res_matched}/{total_count} ({high_res_matched/total_count*100:.2f}%)")
    print(f"低分辨率特征文件匹配: {low_res_matched}/{total_count} ({low_res_matched/total_count*100:.2f}%)")
    
    # 可视化特征
    if args.visualize:
        print("\n可视化特征...")
        visualize_features(high_res_files, low_res_files, args.output_dir, args.csv_path)
    
    print("\n特征验证完成!")

if __name__ == "__main__":
    main()