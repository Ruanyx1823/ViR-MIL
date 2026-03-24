import os
import pandas as pd
import numpy as np
import h5py
from PIL import Image
import cv2
from tqdm import tqdm
import openslide
import warnings
warnings.filterwarnings("ignore")

class JPGSlideImage:
    """
    模拟WSI接口的JPG图像处理类
    将普通JPG图像包装成类似WSI的接口，以便与原有代码兼容
    """
    def __init__(self, path):
        """
        初始化JPGSlideImage对象
        
        参数:
            path (str): JPG图像的路径
        """
        self.path = path
        self.name = os.path.basename(path).split('.')[0]
        self.image = Image.open(path)
        self.width, self.height = self.image.size
        
        # 创建多分辨率金字塔结构
        self.level_dimensions = [
            (self.width, self.height),           # 原始尺寸 (level 0)
            (self.width//2, self.height//2),     # 缩小一半 (level 1)
            (self.width//4, self.height//4)      # 缩小四分之一 (level 2)
        ]
        self.level_downsamples = [1, 2, 4]  # 各级别的下采样因子
        
        # 模拟WSI属性
        self.properties = {
            'aperio.AppMag': '40',  # 模拟40x放大倍率
            'openslide.level-count': '3'  # 3个分辨率级别
        }
    
    def read_region(self, location, level, size):
        """
        模拟WSI的read_region方法，从指定位置和分辨率级别读取图像区域
        
        参数:
            location (tuple): (x, y)坐标，表示要读取区域的左上角
            level (int): 分辨率级别
            size (tuple): (width, height)，要读取区域的大小
            
        返回:
            PIL.Image: 读取的图像区域
        """
        x, y = location
        scale = self.level_downsamples[level]
        x_scaled, y_scaled = x // scale, y // scale
        width, height = size
        
        # 确保坐标不超出图像边界
        x_scaled = max(0, min(x_scaled, self.level_dimensions[level][0] - width))
        y_scaled = max(0, min(y_scaled, self.level_dimensions[level][1] - height))
        
        # 裁剪图像区域
        if level == 0:
            # 对于原始分辨率，直接从原图裁剪
            region = self.image.crop((x_scaled, y_scaled, x_scaled + width, y_scaled + height))
        else:
            # 对于其他分辨率，先缩放原图，再裁剪
            scaled_img = self.image.resize(self.level_dimensions[level], Image.LANCZOS)
            region = scaled_img.crop((x_scaled, y_scaled, x_scaled + width, y_scaled + height))
        
        return region
    
    def get_best_level_for_downsample(self, downsample):
        """
        模拟WSI的get_best_level_for_downsample方法
        
        参数:
            downsample (float): 下采样因子
            
        返回:
            int: 最适合的分辨率级别
        """
        for i, ds in enumerate(self.level_downsamples):
            if ds >= downsample:
                return i
        return len(self.level_downsamples) - 1
    
    def get_thumbnail(self, size):
        """
        获取缩略图
        
        参数:
            size (tuple): (width, height)，缩略图的大小
            
        返回:
            PIL.Image: 缩略图
        """
        return self.image.resize(size, Image.LANCZOS)


def generate_dataset_csv(base_dir='../shujuji', output_path='dataset_csv'):
    """
    生成数据集CSV文件，包含患者ID、幻灯片ID和标签信息
    
    参数:
        base_dir (str): 数据集根目录
        output_path (str): 输出CSV文件的目录
        
    返回:
        pd.DataFrame: 生成的数据集DataFrame
    """
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 初始化数据列表
    csv_data = []
    
    # 处理训练集
    print("处理训练集...")
    for category in ['Stable', 'Developing']:
        folder_path = os.path.join(base_dir, 'train', category)
        files = os.listdir(folder_path)
        for file in tqdm(files):
            if file.endswith('.JPG'):
                patient_id = file.split('_')[0]  # 提取患者ID
                slide_id = file.replace('.JPG', '')  # 使用文件名作为slide_id
                # 添加完整路径以便后续处理
                file_path = os.path.join(folder_path, file)
                csv_data.append([patient_id, slide_id, category, file_path, 'train'])
    
    # 处理测试集
    print("处理测试集...")
    for category in ['Stable', 'Developing']:
        folder_path = os.path.join(base_dir, 'test', category)
        files = os.listdir(folder_path)
        for file in tqdm(files):
            if file.endswith('.JPG'):
                patient_id = file.split('_')[0]  # 提取患者ID
                slide_id = file.replace('.JPG', '')  # 使用文件名作为slide_id
                # 添加完整路径以便后续处理
                file_path = os.path.join(folder_path, file)
                csv_data.append([patient_id, slide_id, category, file_path, 'test'])
    
    # 创建DataFrame
    df = pd.DataFrame(csv_data, columns=['case_id', 'slide_id', 'label', 'file_path', 'split'])
    
    # 保存CSV文件
    csv_file_path = os.path.join(output_path, 'vitiligo_subtyping.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"数据集CSV文件已保存至: {csv_file_path}")
    
    # 统计信息
    print(f"总样本数: {len(df)}")
    print(f"训练集样本数: {len(df[df['split'] == 'train'])}")
    print(f"测试集样本数: {len(df[df['split'] == 'test'])}")
    print(f"Stable类别样本数: {len(df[df['label'] == 'Stable'])}")
    print(f"Developing类别样本数: {len(df[df['label'] == 'Developing'])}")
    
    return df


def is_valid_patch(patch, white_thresh=220, white_percent=0.8, black_thresh=20, black_percent=0.8):
    """
    检查补丁是否有效（不是纯白或纯黑）
    
    参数:
        patch (PIL.Image): 图像补丁
        white_thresh (int): 白色阈值
        white_percent (float): 白色像素占比阈值
        black_thresh (int): 黑色阈值
        black_percent (float): 黑色像素占比阈值
        
    返回:
        bool: 补丁是否有效
    """
    # 转换为numpy数组
    patch_np = np.array(patch)
    
    # 检查是否为纯白
    white_pixels = np.mean(patch_np, axis=2) > white_thresh
    white_ratio = np.sum(white_pixels) / (patch_np.shape[0] * patch_np.shape[1])
    if white_ratio > white_percent:
        return False
    
    # 检查是否为纯黑
    black_pixels = np.mean(patch_np, axis=2) < black_thresh
    black_ratio = np.sum(black_pixels) / (patch_np.shape[0] * patch_np.shape[1])
    if black_ratio > black_percent:
        return False
    
    return True


def process_jpg_to_patches(jpg_path, save_dir, patch_size=256, stride=128):
    """
    从JPG图像提取补丁并保存
    
    参数:
        jpg_path (str): JPG图像路径
        save_dir (str): 保存目录
        patch_size (int): 补丁大小
        stride (int): 滑动窗口步长
        
    返回:
        int: 提取的有效补丁数量
    """
    # 创建JPGSlideImage对象
    slide = JPGSlideImage(jpg_path)
    slide_name = slide.name
    
    # 创建保存目录
    patch_save_dir = os.path.join(save_dir, slide_name)
    os.makedirs(patch_save_dir, exist_ok=True)
    
    # 提取补丁
    coords = []
    width, height = slide.width, slide.height
    
    # 使用滑动窗口提取补丁
    for x in range(0, width - patch_size, stride):
        for y in range(0, height - patch_size, stride):
            # 读取区域
            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
            
            # 过滤无效补丁
            if not is_valid_patch(patch):
                continue
                
            # 保存补丁
            patch_name = f"{x}_{y}.png"
            patch.save(os.path.join(patch_save_dir, patch_name))
            coords.append([x, y])
    
    # 保存坐标信息到h5文件
    h5_path = os.path.join(save_dir, f"{slide_name}.h5")
    with h5py.File(h5_path, 'w') as f:
        if len(coords) > 0:
            f.create_dataset('coords', data=np.array(coords))
            f['coords'].attrs['patch_level'] = 0
            f['coords'].attrs['patch_size'] = patch_size
    
    return len(coords)


def batch_process_slides(csv_path, output_base_dir, patch_size=256, stride=128):
    """
    批量处理所有幻灯片，提取补丁
    
    参数:
        csv_path (str): 数据集CSV文件路径
        output_base_dir (str): 输出基础目录
        patch_size (int): 补丁大小
        stride (int): 滑动窗口步长
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建输出目录
    patches_dir = os.path.join(output_base_dir, f'patches_{patch_size}')
    os.makedirs(patches_dir, exist_ok=True)
    
    # 按分割和类别处理
    for split in ['train', 'test']:
        for category in ['Stable', 'Developing']:
            # 筛选数据
            subset = df[(df['split'] == split) & (df['label'] == category)]
            
            # 创建目录
            category_dir = os.path.join(patches_dir, split, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # 处理每张图像
            print(f"处理 {split} 集的 {category} 类别图像...")
            for _, row in tqdm(subset.iterrows(), total=len(subset)):
                file_path = row['file_path']
                slide_id = row['slide_id']
                
                # 提取补丁
                num_patches = process_jpg_to_patches(
                    file_path, 
                    category_dir, 
                    patch_size=patch_size, 
                    stride=stride
                )
                
                print(f"  - {slide_id}: 提取了 {num_patches} 个有效补丁")


def test_jpg_slide_image():
    """
    测试JPGSlideImage类的功能
    """
    # 选择一个测试图像
    test_image_path = None
    for category in ['Stable', 'Developing']:
        folder_path = os.path.join('../shujuji', 'train', category)
        files = os.listdir(folder_path)
        if files:
            test_image_path = os.path.join(folder_path, files[0])
            break
    
    if not test_image_path:
        print("未找到测试图像")
        return False
    
    try:
        # 创建JPGSlideImage对象
        slide = JPGSlideImage(test_image_path)
        
        # 测试基本属性
        print(f"图像名称: {slide.name}")
        print(f"图像尺寸: {slide.width} x {slide.height}")
        print(f"分辨率级别: {len(slide.level_dimensions)}")
        
        # 测试read_region方法
        patch_size = 256
        patch = slide.read_region((0, 0), 0, (patch_size, patch_size))
        print(f"提取补丁尺寸: {patch.size}")
        
        # 测试不同分辨率级别
        for level in range(len(slide.level_dimensions)):
            patch = slide.read_region((0, 0), level, (patch_size, patch_size))
            print(f"级别 {level} 补丁尺寸: {patch.size}")
        
        print("JPGSlideImage类测试通过")
        return True
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 测试JPGSlideImage类
    print("测试JPGSlideImage类...")
    if test_jpg_slide_image():
        print("JPGSlideImage类测试通过")
    else:
        print("JPGSlideImage类测试失败")
        exit(1)
    
    # 生成数据集CSV文件
    print("\n生成数据集CSV文件...")
    df = generate_dataset_csv()
    
    # 提示用户是否继续处理补丁
    response = input("\n是否继续处理图像补丁? (y/n): ")
    if response.lower() == 'y':
        # 设置参数
        patch_size = 256
        stride = 128
        output_base_dir = 'processed_data'
        
        # 批量处理幻灯片
        print(f"\n开始处理图像补丁 (patch_size={patch_size}, stride={stride})...")
        batch_process_slides(
            'dataset_csv/vitiligo_subtyping.csv',
            output_base_dir,
            patch_size=patch_size,
            stride=stride
        )
        print("图像补丁处理完成")
    else:
        print("跳过图像补丁处理")