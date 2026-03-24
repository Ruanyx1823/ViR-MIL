#!/usr/bin/env python3
"""
检查MURA原始CSV文件格式
"""

import os
import pandas as pd

def check_original_csv():
    """检查原始MURA CSV文件"""
    print("🔍 检查MURA原始CSV文件")
    print("=" * 50)
    
    mura_root = "../MURA-v1.1"
    
    # 检查训练集CSV
    train_csv = os.path.join(mura_root, "train_labeled_studies.csv")
    valid_csv = os.path.join(mura_root, "valid_labeled_studies.csv")
    
    for csv_name, csv_path in [("训练集", train_csv), ("验证集", valid_csv)]:
        print(f"\n📄 {csv_name} CSV: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"❌ 文件不存在")
            continue
            
        try:
            # 读取CSV文件（无表头）
            df = pd.read_csv(csv_path, header=None)
            print(f"✅ 文件读取成功")
            print(f"   行数: {len(df)}")
            print(f"   列数: {len(df.columns)}")
            
            # 显示前几行
            print(f"   前5行内容:")
            for idx, row in df.head(5).iterrows():
                print(f"     {idx + 1}: {row.tolist()}")
                
            # 分析路径格式
            if len(df) > 0:
                print(f"\n   路径格式分析:")
                sample_path = df.iloc[0, 0]  # 第一列第一行
                path_parts = sample_path.split('/')
                print(f"     示例路径: {sample_path}")
                print(f"     路径组成: {path_parts}")
                print(f"     路径层级: {len(path_parts)}")
                
                if len(path_parts) >= 5:
                    print(f"     [0] 根目录: {path_parts[0]}")
                    print(f"     [1] 分割: {path_parts[1]}")
                    print(f"     [2] 身体部位: {path_parts[2]}")
                    print(f"     [3] 患者ID: {path_parts[3]}")
                    print(f"     [4] 研究ID: {path_parts[4]}")
                    if len(path_parts) > 5:
                        print(f"     [5] 图像文件: {path_parts[5]}")
                
        except Exception as e:
            print(f"❌ 读取失败: {e}")

def main():
    """主函数"""
    check_original_csv()
    
    print(f"\n💡 说明:")
    print("MURA原始CSV文件包含完整的图像路径，格式通常是:")
    print("MURA-v1.1/train/XR_ELBOW/patient00001/study1_positive/image1.png")
    print()
    print("我们的CSV生成脚本需要正确解析这些路径来创建:")
    print("- case_id: patient00001")
    print("- slide_id: study1_positive (或完整路径)")
    print("- body_part: XR_ELBOW")
    print("- split: train")

if __name__ == "__main__":
    main()