"""
MURA数据处理模块
将MURA数据集转换为ViLa-MIL框架所需的格式
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import h5py
from PIL import Image
import cv2

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA数据处理')
    parser.add_argument('--mura_root', type=str, default='MURA-v1.1',
                        help='MURA数据集根目录')
    parser.add_argument('--output_dir', type=str, default='dataset_csv',
                        help='输出CSV文件目录')
    parser.add_argument('--body_parts', type=str, default='all',
                        help='处理的身体部位，用逗号分隔，或"all"处理所有')
    parser.add_argument('--processed_dir', type=str, default='processed_data/mura_processed',
                        help='预处理图像输出目录')
    parser.add_argument('--target_size', type=int, default=256,
                        help='目标图像大小')
    return parser.parse_args()

def create_mura_csv(mura_root, output_dir, body_parts='all'):
    """
    创建适合ViLa-MIL的CSV格式数据
    
    参数:
        mura_root: MURA数据集根目录
        output_dir: 输出CSV文件目录
        body_parts: 要处理的身体部位列表或'all'
    
    返回:
        保存的CSV文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像路径数据（包含具体图像文件路径）
    train_images = pd.read_csv(os.path.join(mura_root, 'train_image_paths.csv'), header=None)
    valid_images = pd.read_csv(os.path.join(mura_root, 'valid_image_paths.csv'), header=None)
    
    # 读取标签数据（研究级别的标签）
    train_studies = pd.read_csv(os.path.join(mura_root, 'train_labeled_studies.csv'), header=None)
    valid_studies = pd.read_csv(os.path.join(mura_root, 'valid_labeled_studies.csv'), header=None)
    
    # 重命名列
    train_images.columns = ['image_path']
    valid_images.columns = ['image_path']
    train_studies.columns = ['study_path', 'label']
    valid_studies.columns = ['study_path', 'label']
    
    # 为图像数据添加split列
    train_images['split'] = 'train'
    valid_images['split'] = 'valid'
    
    # 合并图像数据
    all_images = pd.concat([train_images, valid_images], ignore_index=True)
    
    # 为每个图像匹配标签
    # 从图像路径中提取研究路径（去掉图像文件名）
    all_images['study_path'] = all_images['image_path'].apply(lambda x: '/'.join(x.split('/')[:-1]) + '/')
    
    # 合并标签数据
    all_studies = pd.concat([train_studies, valid_studies], ignore_index=True)
    
    # 将图像数据与标签数据合并
    all_data = all_images.merge(all_studies, on='study_path', how='left')
    
    # 提取身体部位、患者ID和研究ID（从图像路径中提取）
    all_data['body_part'] = all_data['image_path'].apply(lambda x: x.split('/')[2])
    all_data['patient_id'] = all_data['image_path'].apply(lambda x: x.split('/')[3])
    all_data['study_id'] = all_data['image_path'].apply(lambda x: x.split('/')[4])
    all_data['image_name'] = all_data['image_path'].apply(lambda x: x.split('/')[-1])
    
    # 如果指定了特定身体部位，进行过滤
    if body_parts != 'all':
        body_part_list = body_parts.split(',')
        all_data = all_data[all_data['body_part'].isin(body_part_list)]
    
    # 创建ViLa-MIL所需的CSV格式
    # slide_id使用完整的图像路径
    vila_df = pd.DataFrame({
        'case_id': all_data['patient_id'],
        'slide_id': all_data['image_path'],  # 使用完整图像路径
        'label': all_data['label'],
        'split': all_data['split'],
        'body_part': all_data['body_part']
    })
    
    # 标签映射: 0=正常, 1=异常
    label_mapping = {0: 'normal', 1: 'abnormal'}
    vila_df['label_name'] = vila_df['label'].map(label_mapping)
    
    # 保存CSV文件
    output_path = os.path.join(output_dir, 'mura_abnormality_detection.csv')
    vila_df.to_csv(output_path, index=False)
    print(f"已创建MURA数据集CSV文件: {output_path}")
    
    # 输出数据集统计信息
    print("\n数据集统计信息:")
    print(f"总样本数: {len(vila_df)}")
    print(f"训练集: {len(vila_df[vila_df['split'] == 'train'])}")
    print(f"验证集: {len(vila_df[vila_df['split'] == 'valid'])}")
    print("\n按身体部位分布:")
    print(vila_df.groupby(['body_part', 'split']).size().unstack())
    print("\n按标签分布:")
    print(vila_df.groupby(['label', 'split']).size().unstack())
    
    return output_path

def preprocess_image(image_path, target_size=(256, 256)):
    """
    预处理X光片图像
    
    参数:
        image_path: 图像路径
        target_size: 目标大小
    
    返回:
        预处理后的图像数组
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 调整大小
    img_resized = cv2.resize(img, target_size)
    
    # 标准化
    img_norm = img_resized / 255.0
    
    # 增强对比度（针对X光片）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_norm = clahe.apply((img_norm * 255).astype(np.uint8))
    img_norm = img_norm / 255.0
    
    return img_norm

def prepare_mura_data_for_feature_extraction(mura_root, output_dir, csv_path, target_size=(256, 256)):
    """
    准备MURA数据用于特征提取
    
    参数:
        mura_root: MURA数据集根目录
        output_dir: 输出目录
        csv_path: 数据集CSV文件路径
        target_size: 目标图像大小
    """
    # 读取数据集CSV
    df = pd.read_csv(csv_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个图像
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理研究"):
        # slide_id现在包含完整的图像路径
        image_path = row['slide_id']
        
        # 移除MURA-v1.1前缀（如果存在）
        if image_path.startswith('MURA-v1.1/'):
            relative_path = image_path[len('MURA-v1.1/'):]
        else:
            relative_path = image_path
            
        # 构建完整图像路径
        full_image_path = os.path.join(mura_root, relative_path)
        
        # 检查图像是否存在
        if not os.path.exists(full_image_path):
            print(f"警告: 路径不存在 {full_image_path}")
            continue
        
        # 为该图像创建输出目录
        image_filename = os.path.basename(relative_path)
        study_name = os.path.basename(os.path.dirname(relative_path))
        output_study_dir = os.path.join(output_dir, row['split'], 
                                        'abnormal' if row['label'] == 1 else 'normal',
                                        f"{row['body_part']}_{row['case_id']}_{study_name}")
        os.makedirs(output_study_dir, exist_ok=True)
        
        # 预处理图像
        try:
            processed_img = preprocess_image(full_image_path, target_size)
            
            # 保存预处理后的图像
            output_img_path = os.path.join(output_study_dir, image_filename)
            cv2.imwrite(output_img_path, (processed_img * 255).astype(np.uint8))
        except Exception as e:
            print(f"处理图像时出错 {full_image_path}: {e}")
    
    print(f"数据准备完成，输出目录: {output_dir}")

def count_images_per_study(mura_root, csv_path):
    """
    统计每个研究的图像数量
    
    参数:
        mura_root: MURA数据集根目录
        csv_path: 数据集CSV文件路径
    """
    # 读取数据集CSV
    df = pd.read_csv(csv_path)
    
    # 按研究分组统计图像数量
    study_groups = df.groupby(['case_id', 'split', 'body_part']).size().reset_index(name='image_count')
    
    # 为每行数据添加图像数量信息
    df = df.merge(study_groups, on=['case_id', 'split', 'body_part'], how='left')
    
    # 获取图像数量列表用于统计（使用合并后的DataFrame）
    image_counts = df['image_count'].tolist()
    
    # 输出统计信息
    print("\n每个研究的图像数量统计:")
    print(f"平均图像数: {np.mean(image_counts):.2f}")
    print(f"中位数图像数: {np.median(image_counts)}")
    print(f"最小图像数: {np.min(image_counts)}")
    print(f"最大图像数: {np.max(image_counts)}")
    
    # 按身体部位统计
    print("\n按身体部位的平均图像数:")
    print(df.groupby('body_part')['image_count'].mean())
    
    # 保存更新后的CSV
    output_path = csv_path.replace('.csv', '_with_image_counts.csv')
    df.to_csv(output_path, index=False)
    print(f"已保存带图像数量的CSV文件: {output_path}")

def main():
    """主函数"""
    args = parse_args()
    
    # 创建CSV文件
    csv_path = create_mura_csv(args.mura_root, args.output_dir, args.body_parts)
    
    # 统计每个研究的图像数量
    count_images_per_study(args.mura_root, csv_path)
    
    # 准备数据用于特征提取
    target_size = (args.target_size, args.target_size)
    prepare_mura_data_for_feature_extraction(args.mura_root, args.processed_dir, csv_path, target_size)
    
    print("MURA数据处理完成!")

if __name__ == "__main__":
    main()