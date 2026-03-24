"""
MURA数据集特征提取模块
从预处理的MURA图像中提取双尺度特征
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import models
import clip
from torch.utils.data import Dataset, DataLoader

class MURAImageDataset(Dataset):
    """MURA图像数据集"""
    
    def __init__(self, image_paths, transform=None):
        """
        初始化
        
        参数:
            image_paths: 图像路径列表
            transform: 图像变换
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 转为RGB以适应预训练模型
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, img_path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA特征提取')
    parser.add_argument('--processed_dir', type=str, default='processed_data/mura_processed',
                        help='预处理图像目录')
    parser.add_argument('--output_dir_l', type=str, default='processed_data/high_res_features',
                        help='高分辨率特征输出目录')
    parser.add_argument('--output_dir_s', type=str, default='processed_data/low_res_features',
                        help='低分辨率特征输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--model_name', type=str, default='clip_RN50',
                        choices=['clip_RN50', 'resnet50_trunc', 'densenet121'],
                        help='特征提取模型')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    return parser.parse_args()

def get_model(model_name, device):
    """
    获取特征提取模型
    
    参数:
        model_name: 模型名称
        device: 设备
    
    返回:
        model: 特征提取模型
        transform: 图像变换
        feature_dim: 特征维度
    """
    if model_name == 'clip_RN50':
        model, preprocess = clip.load('RN50', device)
        feature_dim = 1024
    elif model_name == 'resnet50_trunc':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的FC层
        model = model.to(device)
        model.eval()
        feature_dim = 2048
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的分类器
        model = model.to(device)
        model.eval()
        feature_dim = 1024
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model, preprocess, feature_dim

def extract_features(model, model_name, dataloader, device, feature_dim):
    """
    提取特征
    
    参数:
        model: 特征提取模型
        model_name: 模型名称
        dataloader: 数据加载器
        device: 设备
        feature_dim: 特征维度
    
    返回:
        features: 特征列表
        paths: 图像路径列表
    """
    features = []
    paths = []
    
    with torch.no_grad():
        for images, image_paths in tqdm(dataloader, desc="提取特征"):
            images = images.to(device)
            
            if model_name == 'clip_RN50':
                image_features = model.encode_image(images)
                image_features = image_features.cpu().numpy()
            else:
                image_features = model(images).squeeze()
                image_features = image_features.cpu().numpy()
            
            features.append(image_features)
            paths.extend(image_paths)
    
    if len(features) > 0:
        features = np.vstack(features)
    else:
        features = np.array([])
    
    return features, paths

def create_study_features(features, image_paths, output_dir):
    """
    创建研究级特征
    
    参数:
        features: 图像特征列表
        image_paths: 图像路径列表
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 按研究分组
    study_dict = {}
    coords_dict = {}
    
    for i, path in enumerate(image_paths):
        # 从路径中提取研究信息
        parts = path.split(os.sep)
        split = parts[-4]  # train 或 valid
        label = parts[-3]  # normal 或 abnormal
        study_id = parts[-2]  # {body_part}_{patient_id}_{study_id}
        
        # 创建研究键
        study_key = f"{split}/{label}/{study_id}"
        
        # 添加到字典
        if study_key not in study_dict:
            study_dict[study_key] = []
            coords_dict[study_key] = []
        
        study_dict[study_key].append(features[i])
        # 使用简单的坐标，因为X光片没有明确的空间关系
        coords_dict[study_key].append([i, 0])
    
    # 为每个研究创建特征文件
    for study_key, study_features in tqdm(study_dict.items(), desc="创建研究特征"):
        # 转换为numpy数组
        study_features = np.array(study_features)
        coords = np.array(coords_dict[study_key])
        
        # 创建输出目录
        output_path = os.path.join(output_dir, os.path.dirname(study_key))
        os.makedirs(output_path, exist_ok=True)
        
        # 获取研究ID
        study_id = os.path.basename(study_key)
        
        # 保存为h5文件
        h5_path = os.path.join(output_path, f"{study_id}.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('features', data=study_features)
            f.create_dataset('coords', data=coords)
    
    print(f"已创建 {len(study_dict)} 个研究特征文件")

def process_images(processed_dir, output_dir_l, output_dir_s, model_name, batch_size, device):
    """
    处理图像并提取特征
    
    参数:
        processed_dir: 预处理图像目录
        output_dir_l: 高分辨率特征输出目录
        output_dir_s: 低分辨率特征输出目录
        model_name: 模型名称
        batch_size: 批量大小
        device: 设备
    """
    # 获取模型
    model, preprocess, feature_dim = get_model(model_name, device)
    
    # 获取所有图像路径
    image_paths = []
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建数据集和数据加载器
    dataset = MURAImageDataset(image_paths, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 提取特征
    features, paths = extract_features(model, model_name, dataloader, device, feature_dim)
    
    # 创建高分辨率特征
    create_study_features(features, paths, output_dir_l)
    
    # 创建低分辨率特征（简化版，实际应用中可能需要更复杂的处理）
    # 这里我们使用相同的特征，但在实际应用中，您可能需要对图像进行下采样后再提取特征
    create_study_features(features, paths, output_dir_s)
    
    print("特征提取完成!")

def main():
    """主函数"""
    args = parse_args()
    process_images(args.processed_dir, args.output_dir_l, args.output_dir_s, 
                  args.model_name, args.batch_size, args.device)

if __name__ == "__main__":
    main()