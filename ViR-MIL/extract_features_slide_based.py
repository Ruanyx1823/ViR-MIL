#!/usr/bin/env python3
"""
按slide_id保存特征的特征提取脚本
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import clip
from tqdm import tqdm

class MURADataset(Dataset):
    """MURA数据集类"""
    
    def __init__(self, csv_path, data_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.transform = transform
        
        # 过滤有效样本
        self.valid_samples = []
        for idx, row in self.df.iterrows():
            slide_id = row['slide_id']
            image_path = self.get_image_path(slide_id)
            if image_path and os.path.exists(image_path):
                self.valid_samples.append(idx)
        
        print(f"找到 {len(self.valid_samples)} 个有效样本")
    
    def get_image_path(self, slide_id):
        """获取图像路径"""
        # slide_id应该是完整的相对路径
        if slide_id.startswith('MURA-v1.1/'):
            relative_path = slide_id[len('MURA-v1.1/'):]
        else:
            relative_path = slide_id
        
        full_path = os.path.join(self.data_root, relative_path)
        return full_path if os.path.exists(full_path) else None
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        real_idx = self.valid_samples[idx]
        row = self.df.iloc[real_idx]
        
        slide_id = row['slide_id']
        case_id = row['case_id']
        
        image_path = self.get_image_path(slide_id)
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, slide_id, case_id, image_path

def get_model_and_transform(model_name, device):
    """获取模型和预处理"""
    if model_name == 'clip_RN50':
        model, preprocess = clip.load("RN50", device=device)
        # 使用视觉编码器
        model = model.visual
        return model, preprocess
    else:
        raise ValueError(f"不支持的模型: {model_name}")

def extract_features_batch(model, dataloader, device):
    """批量提取特征"""
    model.eval()
    
    all_features = []
    all_slide_ids = []
    all_case_ids = []
    all_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            images, slide_ids, case_ids, paths = batch
            images = images.to(device)
            
            # 提取特征
            features = model(images)
            
            # 确保特征是2D的
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu().numpy())
            all_slide_ids.extend(slide_ids)
            all_case_ids.extend(case_ids)
            all_paths.extend(paths)
    
    # 合并所有特征
    if len(all_features) == 0:
        print("警告: 没有提取到任何特征")
        return np.array([]), all_slide_ids, all_case_ids, all_paths
    
    all_features = np.vstack(all_features)
    
    return all_features, all_slide_ids, all_case_ids, all_paths

def save_features_by_slide_id(features, slide_ids, case_ids, paths, output_dir):
    """按slide_id保存特征"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"保存 {len(slide_ids)} 个slide的特征...")
    
    for i, slide_id in enumerate(tqdm(slide_ids, desc="保存特征")):
        # 处理slide_id作为文件名（替换路径分隔符）
        safe_slide_id = slide_id.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f"{safe_slide_id}.h5")
        
        with h5py.File(output_path, 'w') as f:
            # 保存单个图像的特征
            f.create_dataset('features', data=features[i:i+1])
            f.create_dataset('coords', data=np.array([[0, 0]]))  # 占位符坐标
            
            # 保存元数据
            f.attrs['slide_id'] = slide_id
            f.attrs['case_id'] = case_ids[i]
            f.attrs['image_path'] = paths[i]
    
    print(f"特征保存完成，输出目录: {output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='按slide_id提取和保存特征')
    parser.add_argument('--data_root', type=str, required=True, help='MURA数据集根目录')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--model_name', type=str, default='clip_RN50', help='模型名称')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    print("开始特征提取...")
    print(f"数据根目录: {args.data_root}")
    print(f"CSV文件: {args.csv_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型: {args.model_name}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载模型
    model, transform = get_model_and_transform(args.model_name, device)
    model = model.to(device)
    
    # 获取特征维度
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = model(dummy_input)
        feature_dim = dummy_output.shape[-1]
    
    print(f"模型加载成功，特征维度: {feature_dim}")
    
    # 创建数据集
    dataset = MURADataset(args.csv_path, args.data_root, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"数据集创建成功，样本数: {len(dataset)}")
    
    # 提取特征
    features, slide_ids, case_ids, paths = extract_features_batch(model, dataloader, device)
    
    if len(features) == 0:
        print("❌ 没有提取到任何特征，请检查数据路径和CSV文件")
        return
    
    print(f"特征提取完成，形状: {features.shape}")
    
    # 保存特征
    save_features_by_slide_id(features, slide_ids, case_ids, paths, args.output_dir)
    
    print("✅ 特征提取流程完成!")

if __name__ == "__main__":
    main()