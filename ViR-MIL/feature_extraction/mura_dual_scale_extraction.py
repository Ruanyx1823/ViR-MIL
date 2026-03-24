"""
MURA数据集双尺度特征提取模块
实现真正的双尺度特征提取，为ViLa-MIL框架提供高低分辨率特征
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
import cv2

class MURADualScaleDataset(Dataset):
    """MURA双尺度图像数据集"""
    
    def __init__(self, image_paths, high_res_size=256, low_res_size=128):
        """
        初始化
        
        参数:
            image_paths: 图像路径列表
            high_res_size: 高分辨率图像大小
            low_res_size: 低分辨率图像大小
        """
        self.image_paths = image_paths
        self.high_res_transform = transforms.Compose([
            transforms.Resize((high_res_size, high_res_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.low_res_transform = transforms.Compose([
            transforms.Resize((low_res_size, low_res_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 应用高低分辨率变换
        high_res_img = self.high_res_transform(image)
        low_res_img = self.low_res_transform(image)
        
        return high_res_img, low_res_img, img_path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA双尺度特征提取')
    parser.add_argument('--processed_dir', type=str, default='processed_data/mura_processed',
                        help='预处理图像目录')
    parser.add_argument('--output_dir_l', type=str, default='processed_data/high_res_features',
                        help='高分辨率特征输出目录')
    parser.add_argument('--output_dir_s', type=str, default='processed_data/low_res_features',
                        help='低分辨率特征输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--model_name', type=str, default='clip_RN50',
                        choices=['clip_RN50', 'resnet50_trunc', 'densenet121', 'medical_resnet'],
                        help='特征提取模型')
    parser.add_argument('--high_res_size', type=int, default=256,
                        help='高分辨率图像大小')
    parser.add_argument('--low_res_size', type=int, default=128,
                        help='低分辨率图像大小')
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
        feature_dim: 特征维度
    """
    if model_name == 'clip_RN50':
        model, _ = clip.load('RN50', device)
        feature_dim = 1024
    elif model_name == 'resnet50_trunc':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的FC层
        model = model.to(device)
        feature_dim = 2048
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的分类器
        model = model.to(device)
        feature_dim = 1024
    elif model_name == 'medical_resnet':
        # 这里可以加载医学图像预训练模型，如MedicalNet
        # 示例使用普通ResNet替代
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model = model.to(device)
        feature_dim = 2048
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    model.eval()
    return model, feature_dim

def extract_dual_scale_features(model, model_name, dataloader, device, feature_dim):
    """
    提取双尺度特征
    
    参数:
        model: 特征提取模型
        model_name: 模型名称
        dataloader: 数据加载器
        device: 设备
        feature_dim: 特征维度
    
    返回:
        high_res_features: 高分辨率特征
        low_res_features: 低分辨率特征
        paths: 图像路径
    """
    high_res_features = []
    low_res_features = []
    paths = []
    
    with torch.no_grad():
        for high_res_imgs, low_res_imgs, image_paths in tqdm(dataloader, desc="提取双尺度特征"):
            high_res_imgs = high_res_imgs.to(device)
            low_res_imgs = low_res_imgs.to(device)
            
            # 提取高分辨率特征
            if model_name == 'clip_RN50':
                high_res_feats = model.encode_image(high_res_imgs)
                high_res_feats = high_res_feats.cpu().numpy()
                
                low_res_feats = model.encode_image(low_res_imgs)
                low_res_feats = low_res_feats.cpu().numpy()
            else:
                high_res_feats = model(high_res_imgs).squeeze()
                high_res_feats = high_res_feats.cpu().numpy()
                
                low_res_feats = model(low_res_imgs).squeeze()
                low_res_feats = low_res_feats.cpu().numpy()
            
            # 处理单样本情况
            if len(high_res_imgs) == 1:
                high_res_feats = high_res_feats.reshape(1, -1)
                low_res_feats = low_res_feats.reshape(1, -1)
            
            high_res_features.append(high_res_feats)
            low_res_features.append(low_res_feats)
            paths.extend(image_paths)
    
    if len(high_res_features) > 0:
        high_res_features = np.vstack(high_res_features)
        low_res_features = np.vstack(low_res_features)
    else:
        high_res_features = np.array([])
        low_res_features = np.array([])
    
    return high_res_features, low_res_features, paths

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
        # 为每个图像创建唯一坐标
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
    return len(study_dict)

def enhance_medical_images(image_paths, output_dir, enhancement_type='clahe'):
    """
    增强医学图像
    
    参数:
        image_paths: 图像路径列表
        output_dir: 输出目录
        enhancement_type: 增强类型
    
    返回:
        enhanced_paths: 增强后的图像路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    enhanced_paths = []
    
    for path in tqdm(image_paths, desc="增强医学图像"):
        # 读取图像
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"警告: 无法读取图像 {path}")
            continue
        
        # 应用增强
        if enhancement_type == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)
        elif enhancement_type == 'adaptive_hist':
            enhanced = cv2.equalizeHist(img)
        elif enhancement_type == 'gamma':
            gamma = 1.2
            enhanced = np.power(img / 255.0, gamma) * 255.0
            enhanced = enhanced.astype(np.uint8)
        else:
            enhanced = img
        
        # 保存增强后的图像
        filename = os.path.basename(path)
        rel_path = os.path.relpath(os.path.dirname(path), start=os.path.dirname(output_dir))
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        output_path = os.path.join(output_subdir, filename)
        cv2.imwrite(output_path, enhanced)
        enhanced_paths.append(output_path)
    
    return enhanced_paths

def process_dual_scale_images(processed_dir, output_dir_l, output_dir_s, model_name, batch_size, 
                             high_res_size, low_res_size, device):
    """
    处理图像并提取双尺度特征
    
    参数:
        processed_dir: 预处理图像目录
        output_dir_l: 高分辨率特征输出目录
        output_dir_s: 低分辨率特征输出目录
        model_name: 模型名称
        batch_size: 批量大小
        high_res_size: 高分辨率图像大小
        low_res_size: 低分辨率图像大小
        device: 设备
    """
    # 获取模型
    model, feature_dim = get_model(model_name, device)
    
    # 获取所有图像路径
    image_paths = []
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建数据集和数据加载器
    dataset = MURADualScaleDataset(image_paths, high_res_size, low_res_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 提取双尺度特征
    high_res_features, low_res_features, paths = extract_dual_scale_features(
        model, model_name, dataloader, device, feature_dim
    )
    
    # 创建研究级特征
    high_res_count = create_study_features(high_res_features, paths, output_dir_l)
    low_res_count = create_study_features(low_res_features, paths, output_dir_s)
    
    print(f"高分辨率特征文件: {high_res_count}")
    print(f"低分辨率特征文件: {low_res_count}")
    print("双尺度特征提取完成!")

def verify_features(high_res_dir, low_res_dir):
    """
    验证提取的特征
    
    参数:
        high_res_dir: 高分辨率特征目录
        low_res_dir: 低分辨率特征目录
    """
    high_res_files = []
    low_res_files = []
    
    # 获取所有特征文件
    for root, dirs, files in os.walk(high_res_dir):
        for file in files:
            if file.endswith('.h5'):
                high_res_files.append(os.path.join(root, file))
    
    for root, dirs, files in os.walk(low_res_dir):
        for file in files:
            if file.endswith('.h5'):
                low_res_files.append(os.path.join(root, file))
    
    print(f"高分辨率特征文件数量: {len(high_res_files)}")
    print(f"低分辨率特征文件数量: {len(low_res_files)}")
    
    # 检查文件是否匹配
    high_res_basenames = set([os.path.basename(f) for f in high_res_files])
    low_res_basenames = set([os.path.basename(f) for f in low_res_files])
    
    if high_res_basenames == low_res_basenames:
        print("高低分辨率特征文件完全匹配")
    else:
        print("警告: 高低分辨率特征文件不匹配")
        missing_high = low_res_basenames - high_res_basenames
        missing_low = high_res_basenames - low_res_basenames
        
        if missing_high:
            print(f"高分辨率缺少 {len(missing_high)} 个文件")
        
        if missing_low:
            print(f"低分辨率缺少 {len(missing_low)} 个文件")
    
    # 检查特征维度
    if high_res_files:
        with h5py.File(high_res_files[0], 'r') as f:
            features = f['features'][:]
            print(f"高分辨率特征维度: {features.shape}")
    
    if low_res_files:
        with h5py.File(low_res_files[0], 'r') as f:
            features = f['features'][:]
            print(f"低分辨率特征维度: {features.shape}")

def main():
    """主函数"""
    args = parse_args()
    
    # 处理图像并提取双尺度特征
    process_dual_scale_images(
        args.processed_dir, args.output_dir_l, args.output_dir_s,
        args.model_name, args.batch_size, args.high_res_size, args.low_res_size,
        args.device
    )
    
    # 验证提取的特征
    verify_features(args.output_dir_l, args.output_dir_s)

if __name__ == "__main__":
    main()