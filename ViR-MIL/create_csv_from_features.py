import os
import pandas as pd
import re
from tqdm import tqdm
import argparse

def parse_filename(filename):
    """
    从 .h5 文件名中解析信息
    示例: MURA-v1.1_train_XR_ELBOW_patient00011_study1_negative_image1.png.h5
    """
    # 正则表达式用于匹配文件名中的各个部分
    pattern = re.compile(
        r"MURA-v1.1_(?P<split>train|valid)_"
        r"(?P<body_part>XR_[A-Z]+)_"
        r"patient(?P<patient_id>\d+)_"
        r"(?P<study_id>study\d+)_"
        r"(?P<label>positive|negative)_"
        r"image(?P<image_num>\d+)\.png\.h5"
    )
    
    match = pattern.match(filename)
    
    if not match:
        # 尝试备用格式 (无 study_id positive/negative)
        pattern_alt = re.compile(
            r"MURA-v1.1_(?P<split>train|valid)_"
            r"(?P<body_part>XR_[A-Z]+)_"
            r"patient(?P<patient_id>\d+)_"
            r"(?P<image_name>image\d+\.png)\.h5"
        )
        match = pattern_alt.match(filename)
        if match:
            # 对于备用格式，需要从路径推断标签
            return match.groupdict()
        return None

    return match.groupdict()

def create_csv_from_h5_features(features_dir, output_csv_path):
    """
    从H5特征文件目录创建CSV
    
    参数:
        features_dir: H5特征文件所在的根目录 (e.g., 'processed_data/low_res_features')
        output_csv_path: 输出CSV文件的路径
    """
    all_files_data = []
    
    # 遍历 train 和 valid 文件夹
    for split in ['train', 'valid']:
        split_path = os.path.join(features_dir, split)
        if not os.path.isdir(split_path):
            print(f"警告: 目录不存在 {split_path}")
            continue
            
        # 遍历 'normal' 和 'abnormal' 文件夹
        for label_name in ['normal', 'abnormal']:
            label_path = os.path.join(split_path, label_name)
            if not os.path.isdir(label_path):
                print(f"警告: 目录不存在 {label_path}")
                continue
            
            # 遍历目录中的所有 .h5 文件
            for filename in tqdm(os.listdir(label_path), desc=f"扫描 {split}/{label_name}"):
                if filename.endswith('.h5'):
                    parsed_info = parse_filename(filename)
                    if parsed_info:
                        # 从解析的信息中构建行
                        case_id = f"patient{parsed_info['patient_id']}"
                        
                        # 确定标签（0 for normal, 1 for abnormal）
                        label = 1 if label_name == 'abnormal' else 0
                        
                        # slide_id 就是文件名本身
                        slide_id = filename
                        
                        all_files_data.append({
                            'case_id': case_id,
                            'slide_id': slide_id,
                            'label': label,
                            'split': split,
                            'body_part': parsed_info['body_part']
                        })
                    else:
                        print(f"警告: 无法解析文件名: {filename}")
    
    if not all_files_data:
        print("错误: 未找到任何有效的 .h5 特征文件。请检查 'features_dir' 路径是否正确。")
        return

    # 创建DataFrame并保存
    df = pd.DataFrame(all_files_data)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_csv(output_csv_path, index=False)
    print(f"\n成功创建CSV文件: {output_csv_path}")
    print(f"总共处理了 {len(df)} 个文件。")

    print("\nCSV文件预览:")
    print(df.head())
    
    print("\n身体部位统计:")
    print(df['body_part'].value_counts())

def main():
    parser = argparse.ArgumentParser(description='从H5特征文件创建数据集CSV')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='H5特征文件所在的根目录 (e.g., processed_data/low_res_features)')
    parser.add_argument('--output_csv', type=str, default='dataset_csv/mura_abnormality_features.csv',
                        help='输出CSV文件的路径')
    args = parser.parse_args()
    
    create_csv_from_h5_features(args.features_dir, args.output_csv)

if __name__ == '__main__':
    main()