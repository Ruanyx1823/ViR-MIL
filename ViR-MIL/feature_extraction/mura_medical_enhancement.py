"""
MURA医学图像增强模块
提供专门针对X光片的图像增强方法
"""

import os
import numpy as np
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage import exposure

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA医学图像增强')
    parser.add_argument('--input_dir', type=str, default='processed_data/mura_processed',
                        help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='processed_data/mura_enhanced',
                        help='增强图像输出目录')
    parser.add_argument('--enhancement_type', type=str, default='clahe',
                        choices=['clahe', 'adaptive_hist', 'gamma', 'contrast', 'combined'],
                        help='增强类型')
    parser.add_argument('--clahe_clip', type=float, default=2.0,
                        help='CLAHE剪裁限制')
    parser.add_argument('--clahe_grid', type=int, default=8,
                        help='CLAHE网格大小')
    parser.add_argument('--gamma', type=float, default=1.2,
                        help='Gamma值')
    return parser.parse_args()

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    应用CLAHE增强
    
    参数:
        image: 输入图像
        clip_limit: 剪裁限制
        tile_grid_size: 网格大小
    
    返回:
        增强后的图像
    """
    # 确保图像为灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(gray)
    
    return enhanced

def apply_adaptive_histogram_equalization(image):
    """
    应用自适应直方图均衡化
    
    参数:
        image: 输入图像
    
    返回:
        增强后的图像
    """
    # 确保图像为灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 应用自适应直方图均衡化
    enhanced = exposure.equalize_adapthist(gray)
    enhanced = (enhanced * 255).astype(np.uint8)
    
    return enhanced

def apply_gamma_correction(image, gamma=1.2):
    """
    应用伽马校正
    
    参数:
        image: 输入图像
        gamma: 伽马值
    
    返回:
        增强后的图像
    """
    # 确保图像为灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 应用伽马校正
    enhanced = np.power(gray / 255.0, gamma) * 255.0
    enhanced = enhanced.astype(np.uint8)
    
    return enhanced

def apply_contrast_enhancement(image):
    """
    应用对比度增强
    
    参数:
        image: 输入图像
    
    返回:
        增强后的图像
    """
    # 确保图像为灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 应用对比度拉伸
    p2, p98 = np.percentile(gray, (2, 98))
    enhanced = exposure.rescale_intensity(gray, in_range=(p2, p98))
    
    return enhanced

def apply_combined_enhancement(image, clahe_clip=2.0, clahe_grid=8, gamma=1.2):
    """
    应用组合增强
    
    参数:
        image: 输入图像
        clahe_clip: CLAHE剪裁限制
        clahe_grid: CLAHE网格大小
        gamma: 伽马值
    
    返回:
        增强后的图像
    """
    # 确保图像为灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 应用伽马校正
    gamma_corrected = np.power(gray / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # 应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gamma_corrected)
    
    # 应用对比度拉伸
    p2, p98 = np.percentile(enhanced, (2, 98))
    enhanced = exposure.rescale_intensity(enhanced, in_range=(p2, p98))
    
    return enhanced

def enhance_image(image_path, output_path, enhancement_type, clahe_clip=2.0, clahe_grid=8, gamma=1.2):
    """
    增强图像
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径
        enhancement_type: 增强类型
        clahe_clip: CLAHE剪裁限制
        clahe_grid: CLAHE网格大小
        gamma: 伽马值
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"警告: 无法读取图像 {image_path}")
        return False
    
    # 应用增强
    if enhancement_type == 'clahe':
        enhanced = apply_clahe(image, clahe_clip, (clahe_grid, clahe_grid))
    elif enhancement_type == 'adaptive_hist':
        enhanced = apply_adaptive_histogram_equalization(image)
    elif enhancement_type == 'gamma':
        enhanced = apply_gamma_correction(image, gamma)
    elif enhancement_type == 'contrast':
        enhanced = apply_contrast_enhancement(image)
    elif enhancement_type == 'combined':
        enhanced = apply_combined_enhancement(image, clahe_clip, clahe_grid, gamma)
    else:
        enhanced = image
    
    # 保存增强后的图像
    cv2.imwrite(output_path, enhanced)
    return True

def enhance_directory(input_dir, output_dir, enhancement_type, clahe_clip=2.0, clahe_grid=8, gamma=1.2):
    """
    增强目录中的所有图像
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        enhancement_type: 增强类型
        clahe_clip: CLAHE剪裁限制
        clahe_grid: CLAHE网格大小
        gamma: 伽马值
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像路径
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 增强每张图像
    success_count = 0
    for path in tqdm(image_paths, desc=f"应用{enhancement_type}增强"):
        # 构建输出路径
        rel_path = os.path.relpath(path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 增强图像
        if enhance_image(path, output_path, enhancement_type, clahe_clip, clahe_grid, gamma):
            success_count += 1
    
    print(f"成功增强 {success_count}/{len(image_paths)} 张图像")
    return success_count

def create_comparison_image(original_path, enhanced_path, output_path):
    """
    创建原始图像和增强图像的对比图
    
    参数:
        original_path: 原始图像路径
        enhanced_path: 增强图像路径
        output_path: 输出图像路径
    """
    # 读取图像
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    enhanced = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None or enhanced is None:
        print(f"警告: 无法读取图像")
        return False
    
    # 调整大小以确保一致
    h, w = original.shape
    enhanced = cv2.resize(enhanced, (w, h))
    
    # 创建对比图
    comparison = np.hstack((original, enhanced))
    
    # 添加文本标签
    comparison = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison, "Enhanced", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 保存对比图
    cv2.imwrite(output_path, comparison)
    return True

def create_enhancement_comparisons(input_dir, enhanced_dir, output_dir, sample_count=10):
    """
    创建增强效果对比图
    
    参数:
        input_dir: 原始图像目录
        enhanced_dir: 增强图像目录
        output_dir: 输出目录
        sample_count: 样本数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像路径
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    # 随机选择样本
    if len(image_paths) > sample_count:
        np.random.seed(42)
        image_paths = np.random.choice(image_paths, sample_count, replace=False)
    
    # 创建对比图
    for i, path in enumerate(image_paths):
        # 构建增强图像路径
        rel_path = os.path.relpath(path, input_dir)
        enhanced_path = os.path.join(enhanced_dir, rel_path)
        
        # 检查增强图像是否存在
        if not os.path.exists(enhanced_path):
            print(f"警告: 增强图像不存在 {enhanced_path}")
            continue
        
        # 创建对比图
        output_path = os.path.join(output_dir, f"comparison_{i}.png")
        create_comparison_image(path, enhanced_path, output_path)
    
    print(f"创建了 {sample_count} 个对比图")

def main():
    """主函数"""
    args = parse_args()
    
    # 增强图像
    enhance_directory(
        args.input_dir, args.output_dir, args.enhancement_type,
        args.clahe_clip, args.clahe_grid, args.gamma
    )
    
    # 创建对比图
    comparison_dir = os.path.join(args.output_dir, "comparisons")
    create_enhancement_comparisons(args.input_dir, args.output_dir, comparison_dir)

if __name__ == "__main__":
    main()