import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from collections import defaultdict
import argparse

def extract_patient_info(base_dir='../shujuji'):
    """
    从shujuji目录中提取患者信息
    
    参数:
        base_dir (str): 数据集根目录
        
    返回:
        tuple: (患者字典, 幻灯片列表)
    """
    print(f"从 {base_dir} 提取患者信息...")
    
    # 初始化患者字典和幻灯片列表
    patients = defaultdict(lambda: {'slides': [], 'label': None, 'split': None})
    slides = []
    
    # 处理训练集和测试集
    for split in ['train', 'test']:
        # 处理稳定期和发展期
        for label in ['Stable', 'Developing']:
            folder_path = os.path.join(base_dir, split, label)
            if not os.path.exists(folder_path):
                print(f"警告: 目录 {folder_path} 不存在")
                continue
                
            files = os.listdir(folder_path)
            for file in files:
                if file.endswith('.JPG'):
                    # 提取患者ID和幻灯片ID
                    patient_id = file.split('_')[0]
                    slide_id = file.replace('.JPG', '')
                    file_path = os.path.join(folder_path, file)
                    
                    # 更新患者信息
                    patients[patient_id]['slides'].append({
                        'slide_id': slide_id,
                        'file_path': file_path,
                        'label': label
                    })
                    patients[patient_id]['label'] = label
                    patients[patient_id]['split'] = split
                    
                    # 添加到幻灯片列表
                    slides.append({
                        'case_id': patient_id,
                        'slide_id': slide_id,
                        'label': label,
                        'file_path': file_path,
                        'split': split
                    })
    
    # 统计信息
    train_patients = sum(1 for p in patients.values() if p['split'] == 'train')
    test_patients = sum(1 for p in patients.values() if p['split'] == 'test')
    stable_patients = sum(1 for p in patients.values() if p['label'] == 'Stable')
    developing_patients = sum(1 for p in patients.values() if p['label'] == 'Developing')
    
    print(f"总患者数: {len(patients)}")
    print(f"训练集患者数: {train_patients}")
    print(f"测试集患者数: {test_patients}")
    print(f"稳定期患者数: {stable_patients}")
    print(f"发展期患者数: {developing_patients}")
    
    print(f"总幻灯片数: {len(slides)}")
    print(f"训练集幻灯片数: {sum(1 for s in slides if s['split'] == 'train')}")
    print(f"测试集幻灯片数: {sum(1 for s in slides if s['split'] == 'test')}")
    print(f"稳定期幻灯片数: {sum(1 for s in slides if s['label'] == 'Stable')}")
    print(f"发展期幻灯片数: {sum(1 for s in slides if s['label'] == 'Developing')}")
    
    return patients, slides

def create_dataset_csv(slides, output_path='dataset_csv/vitiligo_subtyping.csv'):
    """
    创建数据集CSV文件
    
    参数:
        slides (list): 幻灯片列表
        output_path (str): 输出CSV文件路径
        
    返回:
        pd.DataFrame: 数据集DataFrame
    """
    print(f"创建数据集CSV文件: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame(slides)
    
    # 保存CSV文件
    df.to_csv(output_path, index=False)
    print(f"数据集CSV文件已保存至: {output_path}")
    
    return df

def create_patient_level_splits(patients, k=5, val_ratio=0.2, seed=42):
    """
    创建基于患者的数据分割
    
    参数:
        patients (dict): 患者字典
        k (int): 折数
        val_ratio (float): 验证集比例
        seed (int): 随机种子
        
    返回:
        list: k折分割列表
    """
    print(f"创建 {k} 折交叉验证分割...")
    
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    # 分离训练集和测试集患者
    train_patients = {pid: p for pid, p in patients.items() if p['split'] == 'train'}
    test_patients = {pid: p for pid, p in patients.items() if p['split'] == 'test'}
    
    # 按类别分组训练集患者
    stable_train_patients = [pid for pid, p in train_patients.items() if p['label'] == 'Stable']
    developing_train_patients = [pid for pid, p in train_patients.items() if p['label'] == 'Developing']
    
    # 打乱患者顺序
    random.shuffle(stable_train_patients)
    random.shuffle(developing_train_patients)
    
    # 创建k折分割
    splits = []
    for fold in range(k):
        # 计算稳定期验证集大小
        stable_val_size = int(len(stable_train_patients) * val_ratio)
        # 计算发展期验证集大小
        developing_val_size = int(len(developing_train_patients) * val_ratio)
        
        # 计算每折的起始索引
        stable_start = (fold * stable_val_size) % len(stable_train_patients)
        developing_start = (fold * developing_val_size) % len(developing_train_patients)
        
        # 选择验证集患者
        stable_val_patients = stable_train_patients[stable_start:stable_start + stable_val_size]
        developing_val_patients = developing_train_patients[developing_start:developing_start + developing_val_size]
        
        # 合并验证集患者
        val_patients = stable_val_patients + developing_val_patients
        
        # 选择训练集患者(剩余患者)
        train_val_patients = [pid for pid in train_patients.keys() if pid not in val_patients]
        
        # 获取测试集患者
        test_patient_ids = list(test_patients.keys())
        
        # 添加到分割列表
        splits.append({
            'train': train_val_patients,
            'val': val_patients,
            'test': test_patient_ids
        })
        
        # 打印分割信息
        print(f"折 {fold+1}/{k}:")
        print(f"  - 训练集: {len(train_val_patients)} 患者")
        print(f"  - 验证集: {len(val_patients)} 患者 (稳定期: {len(stable_val_patients)}, 发展期: {len(developing_val_patients)})")
        print(f"  - 测试集: {len(test_patient_ids)} 患者")
    
    return splits

def create_slide_level_splits(patients, patient_splits, slides):
    """
    基于患者级分割创建幻灯片级分割
    
    参数:
        patients (dict): 患者字典
        patient_splits (list): 患者级分割列表
        slides (list): 幻灯片列表
        
    返回:
        list: 幻灯片级分割列表
    """
    print("创建幻灯片级分割...")
    
    slide_splits = []
    for fold, split in enumerate(patient_splits):
        # 初始化幻灯片ID列表
        train_slides = []
        val_slides = []
        test_slides = []
        
        # 获取训练集幻灯片
        for pid in split['train']:
            for slide in patients[pid]['slides']:
                train_slides.append(slide['slide_id'])
        
        # 获取验证集幻灯片
        for pid in split['val']:
            for slide in patients[pid]['slides']:
                val_slides.append(slide['slide_id'])
        
        # 获取测试集幻灯片
        for pid in split['test']:
            for slide in patients[pid]['slides']:
                test_slides.append(slide['slide_id'])
        
        # 添加到分割列表
        slide_splits.append({
            'train': train_slides,
            'val': val_slides,
            'test': test_slides
        })
        
        # 打印分割信息
        print(f"折 {fold+1}:")
        print(f"  - 训练集: {len(train_slides)} 幻灯片")
        print(f"  - 验证集: {len(val_slides)} 幻灯片")
        print(f"  - 测试集: {len(test_slides)} 幻灯片")
    
    return slide_splits

def save_splits_to_csv(slide_splits, output_dir='splits/task_vitiligo_subtyping_100'):
    """
    保存分割结果为CSV文件
    
    参数:
        slide_splits (list): 幻灯片级分割列表
        output_dir (str): 输出目录
        
    返回:
        bool: 是否成功保存
    """
    print(f"保存分割结果至 {output_dir}...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每折分割
    for fold, split in enumerate(slide_splits):
        # 创建DataFrame
        max_len = max(len(split['train']), len(split['val']), len(split['test']))
        df = pd.DataFrame({
            'train': pd.Series(split['train'], index=range(len(split['train']))),
            'val': pd.Series(split['val'], index=range(len(split['val']))),
            'test': pd.Series(split['test'], index=range(len(split['test'])))
        })
        
        # 保存CSV文件
        output_path = os.path.join(output_dir, f'splits_{fold}.csv')
        df.to_csv(output_path, index=False)
        print(f"  - 已保存: {output_path}")
    
    return True

def create_splits(base_dir='../shujuji', output_dir='splits/task_vitiligo_subtyping_100', 
                 csv_path='dataset_csv/vitiligo_subtyping.csv', k=5, val_ratio=0.2, seed=42):
    """
    创建数据分割
    
    参数:
        base_dir (str): 数据集根目录
        output_dir (str): 输出目录
        csv_path (str): 数据集CSV文件路径
        k (int): 折数
        val_ratio (float): 验证集比例
        seed (int): 随机种子
        
    返回:
        bool: 是否成功创建
    """
    # 提取患者信息
    patients, slides = extract_patient_info(base_dir)
    
    # 创建数据集CSV文件
    create_dataset_csv(slides, csv_path)
    
    # 创建患者级分割
    patient_splits = create_patient_level_splits(patients, k, val_ratio, seed)
    
    # 创建幻灯片级分割
    slide_splits = create_slide_level_splits(patients, patient_splits, slides)
    
    # 保存分割结果
    save_splits_to_csv(slide_splits, output_dir)
    
    print("数据分割创建完成")
    return True

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='创建白癜风数据集分割')
    parser.add_argument('--base_dir', type=str, default='../shujuji', help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='splits/task_vitiligo_subtyping_100', help='输出目录')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/vitiligo_subtyping.csv', help='数据集CSV文件路径')
    parser.add_argument('--k', type=int, default=5, help='折数')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 创建数据分割
    create_splits(args.base_dir, args.output_dir, args.csv_path, args.k, args.val_ratio, args.seed)