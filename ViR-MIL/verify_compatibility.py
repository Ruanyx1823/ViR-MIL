import os
import sys
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

def check_directory_structure():
    """
    检查目录结构是否符合要求
    
    返回:
        tuple: (是否通过, 错误信息)
    """
    print("检查目录结构...")
    
    # 必要的目录
    required_dirs = [
        'dataset_csv',
        'splits',
        'text_prompt',
        'processed_data'
    ]
    
    # 检查必要的目录是否存在
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        return False, f"缺少必要的目录: {', '.join(missing_dirs)}"
    
    return True, "目录结构检查通过"

def check_csv_files():
    """
    检查CSV文件是否存在和格式是否正确
    
    返回:
        tuple: (是否通过, 错误信息)
    """
    print("检查CSV文件...")
    
    # 检查数据集CSV文件
    csv_path = 'dataset_csv/vitiligo_subtyping.csv'
    if not os.path.exists(csv_path):
        return False, f"未找到数据集CSV文件: {csv_path}"
    
    # 检查CSV文件格式
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['case_id', 'slide_id', 'label', 'file_path', 'split']
        missing_columns = []
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            return False, f"CSV文件缺少必要的列: {', '.join(missing_columns)}"
        
        # 检查标签
        unique_labels = df['label'].unique()
        if not all(label in ['Stable', 'Developing'] for label in unique_labels):
            return False, f"CSV文件包含无效的标签: {unique_labels}"
        
        # 检查分割
        unique_splits = df['split'].unique()
        if not all(split in ['train', 'test'] for split in unique_splits):
            return False, f"CSV文件包含无效的分割: {unique_splits}"
        
        print(f"  - 数据集CSV文件包含 {len(df)} 条记录")
        print(f"  - 标签分布: {df['label'].value_counts().to_dict()}")
        print(f"  - 分割分布: {df['split'].value_counts().to_dict()}")
    
    except Exception as e:
        return False, f"读取CSV文件时出错: {str(e)}"
    
    return True, "CSV文件检查通过"

def check_splits():
    """
    检查分割文件是否存在和格式是否正确
    
    返回:
        tuple: (是否通过, 错误信息)
    """
    print("检查分割文件...")
    
    # 检查分割目录
    split_dir = 'splits/task_vitiligo_subtyping_100'
    if not os.path.exists(split_dir):
        return False, f"未找到分割目录: {split_dir}"
    
    # 检查分割文件
    split_files = [f for f in os.listdir(split_dir) if f.startswith('splits_') and f.endswith('.csv')]
    if not split_files:
        return False, f"在 {split_dir} 中未找到分割文件"
    
    # 检查分割文件格式
    try:
        for split_file in split_files:
            split_path = os.path.join(split_dir, split_file)
            df = pd.read_csv(split_path)
            
            required_columns = ['train', 'val', 'test']
            missing_columns = []
            for col in required_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                return False, f"分割文件 {split_file} 缺少必要的列: {', '.join(missing_columns)}"
            
            print(f"  - 分割文件 {split_file} 包含:")
            print(f"    - 训练集: {df['train'].count()} 幻灯片")
            print(f"    - 验证集: {df['val'].count()} 幻灯片")
            print(f"    - 测试集: {df['test'].count()} 幻灯片")
    
    except Exception as e:
        return False, f"读取分割文件时出错: {str(e)}"
    
    return True, "分割文件检查通过"

def check_text_prompt():
    """
    检查文本提示文件是否存在和格式是否正确
    
    返回:
        tuple: (是否通过, 错误信息)
    """
    print("检查文本提示文件...")
    
    # 检查文本提示文件
    prompt_path = 'text_prompt/vitiligo_two_scale_text_prompt.csv'
    if not os.path.exists(prompt_path):
        return False, f"未找到文本提示文件: {prompt_path}"
    
    # 检查文本提示文件格式
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 去除空行
        lines = [line.strip() for line in lines if line.strip()]
        
        # 检查行数
        if len(lines) != 4:
            return False, f"文本提示文件应包含4行，但实际包含 {len(lines)} 行"
        
        # 检查每行格式
        for i, line in enumerate(lines):
            if not line.startswith('"') or not line.endswith('"'):
                return False, f"文本提示文件第 {i+1} 行格式不正确: {line}"
        
        # 检查内容
        required_keywords = [
            ('stable', 'low'),
            ('developing', 'low'),
            ('stable', 'high'),
            ('developing', 'high')
        ]
        
        for i, (keyword1, keyword2) in enumerate(required_keywords):
            line = lines[i].lower()
            if keyword1 not in line or keyword2 not in line:
                return False, f"文本提示文件第 {i+1} 行缺少关键词 '{keyword1}' 或 '{keyword2}'"
        
        print("  - 文本提示文件格式正确")
        for i, line in enumerate(lines):
            print(f"    - 行 {i+1}: {line[:50]}...")
    
    except Exception as e:
        return False, f"读取文本提示文件时出错: {str(e)}"
    
    return True, "文本提示文件检查通过"

def check_h5_features():
    """
    检查H5特征文件是否存在和格式是否正确
    
    返回:
        tuple: (是否通过, 错误信息)
    """
    print("检查H5特征文件...")
    
    # 检查特征目录
    feature_dirs = [
        'processed_data/low_res_features',
        'processed_data/high_res_features'
    ]
    
    missing_dirs = []
    for dir_path in feature_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        return False, f"缺少特征目录: {', '.join(missing_dirs)}"
    
    # 检查H5文件
    h5_files = []
    for dir_path in feature_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
    
    if not h5_files:
        return False, "未找到H5特征文件"
    
    # 检查H5文件格式
    valid_files = 0
    invalid_files = []
    
    for h5_path in tqdm(h5_files, desc="检查H5文件"):
        try:
            with h5py.File(h5_path, 'r') as f:
                # 检查必要的数据集
                if 'features' not in f:
                    invalid_files.append(f"{h5_path} (缺少'features'数据集)")
                    continue
                
                if 'coords' not in f:
                    invalid_files.append(f"{h5_path} (缺少'coords'数据集)")
                    continue
                
                # 检查数据集形状
                features = f['features'][:]
                coords = f['coords'][:]
                
                if len(features) == 0:
                    invalid_files.append(f"{h5_path} (特征为空)")
                    continue
                
                if len(features) != len(coords):
                    invalid_files.append(f"{h5_path} (特征和坐标数量不匹配)")
                    continue
                
                # 检查特征维度
                if features.shape[1] != 1024:  # CLIP-ResNet50特征维度
                    invalid_files.append(f"{h5_path} (特征维度不正确: {features.shape[1]}, 应为1024)")
                    continue
                
                valid_files += 1
        
        except Exception as e:
            invalid_files.append(f"{h5_path} (错误: {str(e)})")
    
    if invalid_files:
        print(f"  - 发现 {len(invalid_files)} 个无效的H5文件:")
        for file in invalid_files[:10]:  # 只显示前10个
            print(f"    - {file}")
        if len(invalid_files) > 10:
            print(f"    - ... 还有 {len(invalid_files) - 10} 个")
    
    print(f"  - 有效H5文件: {valid_files}/{len(h5_files)}")
    
    if valid_files == 0:
        return False, "未找到有效的H5特征文件"
    
    return True, f"H5特征文件检查通过 (有效: {valid_files}, 无效: {len(invalid_files)})"

def check_main_patch():
    """
    检查main_patch.py文件是否存在
    
    返回:
        tuple: (是否通过, 错误信息)
    """
    print("检查main_patch.py文件...")
    
    # 检查main_patch.py文件
    if not os.path.exists('main_patch.py'):
        return False, "未找到main_patch.py文件"
    
    print("  - main_patch.py文件存在")
    return True, "main_patch.py文件检查通过"

def generate_run_command():
    """
    生成运行ViLa-MIL模型的命令
    
    返回:
        str: 运行命令
    """
    command = """
python main.py \\
--seed 1 \\
--drop_out \\
--early_stopping \\
--lr 1e-4 \\
--k 5 \\
--label_frac 1 \\
--bag_loss ce \\
--task 'task_vitiligo_subtyping' \\
--results_dir './results' \\
--exp_code 'vitiligo_exp' \\
--model_type ViLa_MIL \\
--mode transformer \\
--log_data \\
--data_root_dir './processed_data' \\
--data_folder_s 'low_res_features' \\
--data_folder_l 'high_res_features' \\
--split_dir 'task_vitiligo_subtyping_100' \\
--text_prompt_path './text_prompt/vitiligo_two_scale_text_prompt.csv' \\
--prototype_number 16
"""
    return command

def verify_all():
    """
    验证所有兼容性检查
    
    返回:
        bool: 是否全部通过
    """
    checks = [
        check_directory_structure,
        check_csv_files,
        check_splits,
        check_text_prompt,
        check_h5_features,
        check_main_patch
    ]
    
    all_passed = True
    results = []
    
    for check_func in checks:
        passed, message = check_func()
        results.append((passed, message))
        if not passed:
            all_passed = False
    
    print("\n验证结果:")
    for i, (passed, message) in enumerate(results):
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {message}")
    
    if all_passed:
        print("\n所有验证通过，可以运行ViLa-MIL模型")
        print("\n运行命令:")
        print(generate_run_command())
    else:
        print("\n验证失败，请修复上述问题后再运行ViLa-MIL模型")
    
    return all_passed

if __name__ == "__main__":
    verify_all()