"""
MURA特征提取工具模块
提供辅助函数用于MURA数据集的特征提取
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, normalize

class MURADualScaleTransform:
    """MURA双尺度图像变换"""
    
    def __init__(self, high_res_size=256, low_res_size=128):
        """
        初始化
        
        参数:
            high_res_size: 高分辨率大小
            low_res_size: 低分辨率大小
        """
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
    
    def __call__(self, image):
        """
        应用变换
        
        参数:
            image: PIL图像
        
        返回:
            high_res: 高分辨率图像
            low_res: 低分辨率图像
        """
        high_res = self.high_res_transform(image)
        low_res = self.low_res_transform(image)
        
        return high_res, low_res

def enhance_xray_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    增强X光片对比度
    
    参数:
        image: PIL图像
        clip_limit: CLAHE剪裁限制
        tile_grid_size: CLAHE网格大小
    
    返回:
        增强后的PIL图像
    """
    import cv2
    
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 如果是RGB图像，转换为灰度
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_array)
    
    # 转回PIL图像
    return Image.fromarray(enhanced)

def create_dual_scale_batch(image_paths, transform=None, device='cuda'):
    """
    创建双尺度批次
    
    参数:
        image_paths: 图像路径列表
        transform: 图像变换
        device: 设备
    
    返回:
        high_res_batch: 高分辨率批次
        low_res_batch: 低分辨率批次
    """
    high_res_images = []
    low_res_images = []
    
    for path in image_paths:
        # 读取图像
        image = Image.open(path).convert('RGB')
        
        # 应用变换
        if transform:
            high_res, low_res = transform(image)
            high_res_images.append(high_res)
            low_res_images.append(low_res)
    
    # 堆叠为批次
    if high_res_images:
        high_res_batch = torch.stack(high_res_images).to(device)
        low_res_batch = torch.stack(low_res_images).to(device)
    else:
        high_res_batch = torch.tensor([]).to(device)
        low_res_batch = torch.tensor([]).to(device)
    
    return high_res_batch, low_res_batch

def get_study_image_paths(processed_dir):
    """
    获取按研究分组的图像路径
    
    参数:
        processed_dir: 预处理图像目录
    
    返回:
        study_images: 按研究分组的图像路径字典
    """
    study_images = {}
    
    for root, dirs, files in os.walk(processed_dir):
        image_files = [f for f in files if f.endswith('.png')]
        
        if image_files:
            # 提取研究信息
            parts = root.split(os.sep)
            if len(parts) >= 3:
                split = parts[-3]  # train 或 valid
                label = parts[-2]  # normal 或 abnormal
                study_id = parts[-1]  # {body_part}_{patient_id}_{study_id}
                
                # 创建研究键
                study_key = f"{split}/{label}/{study_id}"
                
                # 添加图像路径
                study_images[study_key] = [os.path.join(root, f) for f in image_files]
    
    return study_images

def extract_coords_from_paths(image_paths):
    """
    从图像路径中提取坐标信息
    
    参数:
        image_paths: 图像路径列表
    
    返回:
        coords: 坐标数组
    """
    coords = []
    
    for i, path in enumerate(image_paths):
        # 对于X光片，我们使用简单的索引作为坐标
        # 在实际应用中，您可能需要更复杂的坐标提取
        coords.append([i, 0])
    
    return np.array(coords)

def check_feature_files(output_dir, expected_studies):
    """
    检查特征文件是否完整
    
    参数:
        output_dir: 特征输出目录
        expected_studies: 预期的研究数量
    
    返回:
        is_complete: 是否完整
        missing_studies: 缺失的研究列表
    """
    # 计数已创建的特征文件
    created_files = 0
    study_files = []
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.h5'):
                created_files += 1
                study_files.append(os.path.join(root, file))
    
    is_complete = (created_files == expected_studies)
    missing_studies = expected_studies - created_files if expected_studies > created_files else 0
    
    return is_complete, missing_studies, study_files