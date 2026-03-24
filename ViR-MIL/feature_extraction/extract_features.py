#!/usr/bin/env python3
"""
通用MURA特征提取脚本
支持从原始MURA数据集直接提取特征，无需预处理
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import h5py
from tqdm import tqdm
import clip

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MURADataset(Dataset):
    """MURA数据集类"""
    
    def __init__(self, csv_path, data_root, transform=None, patch_size=512):
        """
        初始化数据集
        
        参数:
            csv_path: CSV文件路径
            data_root: MURA数据根目录
            transform: 图像变换
            patch_size: 补丁大小
        """
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.transform = transform
        self.patch_size = patch_size
        
        print(f"CSV文件包含 {len(self.df)} 行数据")
        print(f"CSV文件列名: {list(self.df.columns)}")
        if len(self.df) > 0:
            print(f"前3行slide_id示例: {self.df['slide_id'].head(3).tolist()}")
        
        # 过滤存在的图像文件
        valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = self.get_image_path(row)
            if os.path.exists(img_path):
                valid_indices.append(idx)
            elif idx < 5:  # 只打印前5个失败的路径
                print(f"文件不存在: {img_path}")
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"找到 {len(self.df)} 个有效样本")
    
    def get_image_path(self, row):
        """根据行信息构建图像路径"""
        # slide_id现在直接包含完整的图像文件路径
        # 例如: MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image1.png
        slide_id = row['slide_id']
        
        # 方法1: 如果slide_id包含MURA-v1.1前缀，去掉它
        if slide_id.startswith('MURA-v1.1/'):
            relative_path = slide_id[len('MURA-v1.1/'):]
            img_path = os.path.join(self.data_root, relative_path)
            if os.path.exists(img_path):
                return img_path
        
        # 方法2: 直接使用slide_id作为相对路径
        img_path = os.path.join(self.data_root, slide_id)
        if os.path.exists(img_path):
            return img_path
        
        # 如果都不存在，打印调试信息
        print(f"警告: 找不到图像文件")
        print(f"  slide_id: {slide_id}")
        print(f"  尝试的路径:")
        if slide_id.startswith('MURA-v1.1/'):
            relative_path = slide_id[len('MURA-v1.1/'):]
            print(f"    方法1: {os.path.join(self.data_root, relative_path)}")
        print(f"    方法2: {os.path.join(self.data_root, slide_id)}")
        
        # 返回第一个尝试的路径作为默认值
        if slide_id.startswith('MURA-v1.1/'):
            relative_path = slide_id[len('MURA-v1.1/'):]
            return os.path.join(self.data_root, relative_path)
        else:
            return os.path.join(self.data_root, slide_id)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.get_image_path(row)
        
        try:
            # 读取图像
            image = Image.open(img_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, img_path, row['case_id'], row['slide_id']
        
        except Exception as e:
            print(f"读取图像失败 {img_path}: {e}")
            # 返回一个默认的黑色图像
            if self.transform:
                default_image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                default_image = torch.zeros(3, 224, 224)
            return default_image, img_path, row['case_id'], row['slide_id']

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA特征提取')
    parser.add_argument('--data_root', type=str, required=True,
                        help='MURA数据集根目录')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='数据集CSV文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='特征输出目录')
    parser.add_argument('--patch_size', type=int, default=512,
                        help='补丁大小')
    parser.add_argument('--step_size', type=int, default=256,
                        help='步长大小')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--model_name', type=str, default='clip_RN50',
                        choices=['clip_RN50', 'resnet50', 'densenet121'],
                        help='特征提取模型 (推荐使用clip_RN50，与ViLa-MIL原版一致)')
    parser.add_argument('--preset', type=str, default=None,
                        help='预设配置文件路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    return parser.parse_args()

def get_model_and_transform(model_name, device):
    """获取模型和变换"""
    if model_name == 'clip_RN50':
        model, preprocess = clip.load("RN50", device=device)
        # 使用CLIP的视觉编码器
        model = model.visual
        model.eval()
        feature_dim = 2048
        return model, preprocess, feature_dim
    
    elif model_name == 'resnet50':
        from torchvision import models
        model = models.resnet50(pretrained=True)
        # 移除最后的分类层
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(device)
        model.eval()
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        feature_dim = 2048
        return model, preprocess, feature_dim
    
    elif model_name == 'densenet121':
        from torchvision import models
        model = models.densenet121(pretrained=True)
        # 移除最后的分类层
        model.classifier = nn.Identity()
        model = model.to(device)
        model.eval()
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        feature_dim = 1024
        return model, preprocess, feature_dim
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")

def extract_features_batch(model, dataloader, device, feature_dim):
    """批量提取特征"""
    all_features = []
    all_paths = []
    all_case_ids = []
    all_slide_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, paths, case_ids, slide_ids) in enumerate(tqdm(dataloader, desc="提取特征")):
            images = images.to(device)
            
            # 提取特征
            features = model(images)
            
            # 确保特征是2D的
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu().numpy())
            all_paths.extend(paths)
            all_case_ids.extend(case_ids)
            all_slide_ids.extend(slide_ids)
    
    # 合并所有特征
    if len(all_features) == 0:
        print("警告: 没有提取到任何特征")
        return np.array([]), all_paths, all_case_ids, all_slide_ids
    
    all_features = np.vstack(all_features)
    
    return all_features, all_paths, all_case_ids, all_slide_ids

def save_features_by_case(features, paths, case_ids, slide_ids, output_dir):
    """按case_id保存特征"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 按case_id分组
    case_dict = {}
    for i, case_id in enumerate(case_ids):
        if case_id not in case_dict:
            case_dict[case_id] = {
                'features': [],
                'coords': [],
                'paths': [],
                'slide_ids': []
            }
        
        case_dict[case_id]['features'].append(features[i])
        case_dict[case_id]['coords'].append([0, 0])  # 占位符坐标
        case_dict[case_id]['paths'].append(paths[i])
        case_dict[case_id]['slide_ids'].append(slide_ids[i])
    
    print(f"保存 {len(case_dict)} 个case的特征...")
    
    for case_id, data in tqdm(case_dict.items(), desc="保存特征"):
        output_path = os.path.join(output_dir, f"{case_id}.h5")
        
        with h5py.File(output_path, 'w') as f:
            # 保存特征
            features_array = np.array(data['features'])
            f.create_dataset('features', data=features_array)
            
            # 保存坐标
            coords_array = np.array(data['coords'])
            f.create_dataset('coords', data=coords_array)
            
            # 保存元数据
            f.attrs['case_id'] = case_id
            f.attrs['num_patches'] = len(data['features'])
            f.attrs['feature_dim'] = features_array.shape[1]
            
            # 保存路径信息（如果需要）
            paths_str = [p.encode('utf-8') for p in data['paths']]
            f.create_dataset('paths', data=paths_str)

def main():
    """主函数"""
    args = parse_args()
    
    print(f"开始特征提取...")
    print(f"数据根目录: {args.data_root}")
    print(f"CSV文件: {args.csv_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型: {args.model_name}")
    print(f"设备: {args.device}")
    
    # 检查输入文件
    if not os.path.exists(args.csv_path):
        print(f"错误: CSV文件不存在: {args.csv_path}")
        return
    
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录不存在: {args.data_root}")
        return
    
    # 获取模型和变换
    try:
        model, preprocess, feature_dim = get_model_and_transform(args.model_name, args.device)
        print(f"模型加载成功，特征维度: {feature_dim}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建数据集
    try:
        dataset = MURADataset(args.csv_path, args.data_root, transform=preprocess, 
                             patch_size=args.patch_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True)
        print(f"数据集创建成功，样本数: {len(dataset)}")
    except Exception as e:
        print(f"数据集创建失败: {e}")
        return
    
    # 提取特征
    try:
        features, paths, case_ids, slide_ids = extract_features_batch(
            model, dataloader, args.device, feature_dim)
        
        if len(features) == 0:
            print("❌ 没有提取到任何特征，请检查数据路径和CSV文件")
            return
            
        print(f"特征提取完成，形状: {features.shape}")
    except Exception as e:
        print(f"特征提取失败: {e}")
        return
    
    # 保存特征
    try:
        save_features_by_case(features, paths, case_ids, slide_ids, args.output_dir)
        print(f"特征保存完成，输出目录: {args.output_dir}")
    except Exception as e:
        print(f"特征保存失败: {e}")
        return
    
    print("✅ 特征提取流程完成!")

if __name__ == "__main__":
    main()