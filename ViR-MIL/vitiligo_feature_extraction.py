import os
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing
import numpy as np
import h5py
from PIL import Image, ImageFile
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
import clip
from concurrent.futures import ThreadPoolExecutor

# 防止PIL截断图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 设置多进程共享策略
torch.multiprocessing.set_sharing_strategy('file_system')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_transforms_clip(pretrained=False):
    """
    获取CLIP模型的图像转换
    
    参数:
        pretrained (bool): 是否使用预训练模型的均值和标准差
        
    返回:
        torchvision.transforms: 图像转换函数
    """
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    
    trnsfrms_val = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std)
    ])
    return trnsfrms_val


class PatchDataset(torch.utils.data.Dataset):
    """
    用于加载图像补丁的数据集类
    """
    def __init__(self, patch_dir, transform=None):
        """
        初始化补丁数据集
        
        参数:
            patch_dir (str): 补丁目录
            transform (callable, optional): 图像转换函数
        """
        self.patch_dir = patch_dir
        self.transform = transform
        self.patch_files = [f for f in os.listdir(patch_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        patch_file = self.patch_files[idx]
        patch_path = os.path.join(self.patch_dir, patch_file)
        
        # 加载图像
        image = Image.open(patch_path).convert('RGB')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return image, patch_file


def extract_features_from_patches(patch_dir, model, batch_size=32):
    """
    从补丁中提取特征
    
    参数:
        patch_dir (str): 补丁目录
        model: 特征提取模型
        batch_size (int): 批处理大小
        
    返回:
        tuple: (特征, 文件名)
    """
    # 创建数据集和数据加载器
    transform = eval_transforms_clip(pretrained=True)
    dataset = PatchDataset(patch_dir, transform)
    
    if len(dataset) == 0:
        print(f"警告: {patch_dir} 中没有找到补丁")
        return None, None
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 提取特征
    all_features = []
    all_filenames = []
    
    with torch.no_grad():
        for batch, filenames in tqdm(dataloader, desc=f"提取特征 ({patch_dir})"):
            batch = batch.to(device)
            # 使用CLIP模型提取特征
            features = model.encode_image(batch).cpu().numpy()
            all_features.append(features)
            all_filenames.extend(filenames)
    
    if all_features:
        all_features = np.vstack(all_features)
        return all_features, all_filenames
    else:
        return None, None


def save_features_to_h5(features, filenames, output_path):
    """
    将特征保存到H5文件
    
    参数:
        features (numpy.ndarray): 特征数组
        filenames (list): 文件名列表
        output_path (str): 输出文件路径
    """
    if features is None or filenames is None:
        print(f"警告: 没有特征可保存到 {output_path}")
        return
    
    # 提取坐标信息
    coords = []
    for filename in filenames:
        # 从文件名中提取坐标 (格式: x_y.png)
        x, y = filename.rstrip('.png').split('_')
        coords.append([int(x), int(y)])
    
    # 保存到H5文件
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('features', data=features)
        f.create_dataset('coords', data=np.array(coords))
    
    print(f"特征已保存到 {output_path}")


def process_slide(slide_id, patches_dir, output_dir, model, resolution='low'):
    """
    处理单个幻灯片的补丁，提取特征并保存
    
    参数:
        slide_id (str): 幻灯片ID
        patches_dir (str): 补丁目录
        output_dir (str): 输出目录
        model: 特征提取模型
        resolution (str): 分辨率类型 ('low' 或 'high')
    """
    # 构建补丁目录路径
    patch_dir = os.path.join(patches_dir, slide_id)
    
    # 检查目录是否存在
    if not os.path.exists(patch_dir):
        print(f"警告: 未找到补丁目录 {patch_dir}")
        return
    
    # 提取特征
    features, filenames = extract_features_from_patches(patch_dir, model)
    
    # 如果没有提取到特征，返回
    if features is None:
        return
    
    # 保存特征
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{slide_id}.h5")
    save_features_to_h5(features, filenames, output_path)


def batch_process_slides(csv_path, patches_base_dir, output_base_dir, model_name="RN50"):
    """
    批量处理所有幻灯片，提取特征
    
    参数:
        csv_path (str): 数据集CSV文件路径
        patches_base_dir (str): 补丁基础目录
        output_base_dir (str): 输出基础目录
        model_name (str): 模型名称
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 加载CLIP模型
    print(f"加载 CLIP {model_name} 模型...")
    model, _ = clip.load(model_name, device=device)
    
    # 创建输出目录
    low_res_dir = os.path.join(output_base_dir, 'low_res_features')
    high_res_dir = os.path.join(output_base_dir, 'high_res_features')
    os.makedirs(low_res_dir, exist_ok=True)
    os.makedirs(high_res_dir, exist_ok=True)
    
    # 获取所有幻灯片ID
    slide_ids = df['slide_id'].unique()
    
    # 处理每个幻灯片
    print(f"开始处理 {len(slide_ids)} 个幻灯片...")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=4) as executor:
        for split in ['train', 'test']:
            for category in ['Stable', 'Developing']:
                # 筛选数据
                subset = df[(df['split'] == split) & (df['label'] == category)]
                subset_slide_ids = subset['slide_id'].unique()
                
                # 补丁目录
                patches_dir = os.path.join(patches_base_dir, f'patches_256/{split}/{category}')
                
                # 输出目录
                low_res_output_dir = os.path.join(low_res_dir, f'{split}/{category}')
                high_res_output_dir = os.path.join(high_res_dir, f'{split}/{category}')
                os.makedirs(low_res_output_dir, exist_ok=True)
                os.makedirs(high_res_output_dir, exist_ok=True)
                
                print(f"处理 {split} 集的 {category} 类别 ({len(subset_slide_ids)} 个幻灯片)...")
                
                # 提交任务到线程池
                for slide_id in subset_slide_ids:
                    # 我们将同一个补丁集合用于低分辨率和高分辨率特征提取
                    # 在实际应用中，可能需要为不同分辨率准备不同的补丁
                    executor.submit(
                        process_slide, 
                        slide_id, 
                        patches_dir, 
                        low_res_output_dir, 
                        model, 
                        'low'
                    )
                    
                    executor.submit(
                        process_slide, 
                        slide_id, 
                        patches_dir, 
                        high_res_output_dir, 
                        model, 
                        'high'
                    )
    
    print("特征提取完成")


def test_feature_extraction():
    """
    测试特征提取功能
    """
    # 选择一个测试目录
    test_dir = None
    for split in ['train', 'test']:
        for category in ['Stable', 'Developing']:
            patches_dir = os.path.join('processed_data', f'patches_256/{split}/{category}')
            if os.path.exists(patches_dir):
                subdirs = [d for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))]
                if subdirs:
                    test_dir = os.path.join(patches_dir, subdirs[0])
                    break
        if test_dir:
            break
    
    if not test_dir:
        print("未找到测试目录")
        return False
    
    try:
        # 加载CLIP模型
        model, _ = clip.load("RN50", device=device)
        
        # 提取特征
        features, filenames = extract_features_from_patches(test_dir, model, batch_size=4)
        
        if features is None:
            print(f"未能从 {test_dir} 提取特征")
            return False
        
        print(f"成功提取特征: {features.shape}")
        
        # 测试保存到H5文件
        test_output = "test_features.h5"
        save_features_to_h5(features, filenames, test_output)
        
        # 验证H5文件
        with h5py.File(test_output, 'r') as f:
            saved_features = f['features'][:]
            saved_coords = f['coords'][:]
            
            print(f"保存的特征形状: {saved_features.shape}")
            print(f"保存的坐标形状: {saved_coords.shape}")
            
            # 清理测试文件
            os.remove(test_output)
        
        print("特征提取测试通过")
        return True
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从补丁中提取特征')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/vitiligo_subtyping.csv',
                        help='数据集CSV文件路径')
    parser.add_argument('--patches_dir', type=str, default='processed_data',
                        help='补丁基础目录')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='输出基础目录')
    parser.add_argument('--model_name', type=str, default='RN50',
                        help='CLIP模型名称 (RN50, ViT-B/32, etc.)')
    parser.add_argument('--test', action='store_true',
                        help='运行测试')
    
    args = parser.parse_args()
    
    if args.test:
        # 运行测试
        print("测试特征提取功能...")
        if test_feature_extraction():
            print("特征提取测试通过")
        else:
            print("特征提取测试失败")
            exit(1)
    else:
        # 批量处理幻灯片
        print(f"开始批量处理幻灯片...")
        batch_process_slides(
            args.csv_path,
            args.patches_dir,
            args.output_dir,
            args.model_name
        )