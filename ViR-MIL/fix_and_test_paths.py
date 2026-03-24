#!/usr/bin/env python3
"""
修复路径问题并测试
"""

import os
import pandas as pd
import subprocess
import sys

def step1_check_original_csv():
    """步骤1: 检查原始MURA CSV文件"""
    print("🔍 步骤1: 检查原始MURA CSV文件")
    print("=" * 50)
    
    mura_root = "../MURA-v1.1"
    train_csv = os.path.join(mura_root, "train_labeled_studies.csv")
    
    if not os.path.exists(train_csv):
        print(f"❌ 原始CSV文件不存在: {train_csv}")
        print("请确保MURA数据集已正确下载")
        return False
    
    try:
        df = pd.read_csv(train_csv, header=None)
        print(f"✅ 原始CSV文件读取成功，包含 {len(df)} 行")
        
        # 显示前3行
        print("前3行示例:")
        for idx, row in df.head(3).iterrows():
            path = row[0]
            label = row[1]
            print(f"  {idx + 1}: {path} -> 标签: {label}")
        
        return True
    except Exception as e:
        print(f"❌ 读取原始CSV失败: {e}")
        return False

def step2_regenerate_csv():
    """步骤2: 重新生成CSV文件"""
    print(f"\n🔄 步骤2: 重新生成CSV文件")
    print("=" * 50)
    
    # 删除旧的CSV文件
    old_csv = "dataset_csv/mura_abnormality_detection.csv"
    if os.path.exists(old_csv):
        os.remove(old_csv)
        print(f"✅ 删除旧CSV文件: {old_csv}")
    
    # 重新生成CSV
    cmd = [
        sys.executable, 
        "mura_data_processing.py",
        "--mura_root", "../MURA-v1.1",
        "--output_dir", "dataset_csv",
        "--body_parts", "all"
    ]
    
    try:
        print("执行命令:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ CSV文件重新生成成功")
            print(result.stdout)
        else:
            print("❌ CSV文件生成失败")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 执行命令失败: {e}")
        return False
    
    return True

def step3_check_new_csv():
    """步骤3: 检查新生成的CSV文件"""
    print(f"\n📄 步骤3: 检查新生成的CSV文件")
    print("=" * 50)
    
    csv_path = "dataset_csv/mura_abnormality_detection.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 新CSV文件不存在: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 新CSV文件读取成功，包含 {len(df)} 行")
        print(f"列名: {list(df.columns)}")
        
        # 显示前3行
        print("前3行示例:")
        for idx, row in df.head(3).iterrows():
            print(f"  {idx + 1}:")
            print(f"    case_id: {row['case_id']}")
            print(f"    slide_id: {row['slide_id']}")
            print(f"    body_part: {row['body_part']}")
            print(f"    split: {row['split']}")
            print(f"    label: {row['label']}")
        
        return True
    except Exception as e:
        print(f"❌ 读取新CSV失败: {e}")
        return False

def step4_test_path_construction():
    """步骤4: 测试路径构建"""
    print(f"\n🛠️ 步骤4: 测试路径构建")
    print("=" * 50)
    
    csv_path = "dataset_csv/mura_abnormality_detection.csv"
    data_root = "../MURA-v1.1"
    
    try:
        df = pd.read_csv(csv_path)
        
        # 测试前5行的路径构建
        success_count = 0
        for idx, row in df.head(5).iterrows():
            slide_id = row['slide_id']
            
            print(f"\n测试行 {idx + 1}:")
            print(f"  slide_id: {slide_id}")
            
            # 方法1: 直接使用slide_id
            path1 = os.path.join(data_root, slide_id)
            exists1 = os.path.exists(path1)
            print(f"  方法1: {path1} -> {'✅' if exists1 else '❌'}")
            
            # 方法2: 去掉MURA-v1.1前缀
            if slide_id.startswith('MURA-v1.1/'):
                relative_path = slide_id[len('MURA-v1.1/'):]
                path2 = os.path.join(data_root, relative_path)
                exists2 = os.path.exists(path2)
                print(f"  方法2: {path2} -> {'✅' if exists2 else '❌'}")
                
                if exists2:
                    success_count += 1
            elif exists1:
                success_count += 1
        
        print(f"\n路径构建成功率: {success_count}/5")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 路径构建测试失败: {e}")
        return False

def step5_test_feature_extraction():
    """步骤5: 测试特征提取"""
    print(f"\n🚀 步骤5: 测试特征提取")
    print("=" * 50)
    
    cmd = [
        sys.executable,
        "feature_extraction/extract_features.py",
        "--data_root", "../MURA-v1.1",
        "--csv_path", "dataset_csv/mura_abnormality_detection.csv",
        "--output_dir", "processed_data/test_features",
        "--batch_size", "4",
        "--model_name", "clip_RN50"
    ]
    
    try:
        print("执行测试命令:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 特征提取测试成功")
            return True
        else:
            print("❌ 特征提取测试失败")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 特征提取测试超时")
        return False
    except Exception as e:
        print(f"❌ 特征提取测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 MURA路径问题修复和测试")
    print("=" * 60)
    
    steps = [
        ("检查原始CSV", step1_check_original_csv),
        ("重新生成CSV", step2_regenerate_csv),
        ("检查新CSV", step3_check_new_csv),
        ("测试路径构建", step4_test_path_construction),
        ("测试特征提取", step5_test_feature_extraction),
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
            
            if not success:
                print(f"\n❌ {step_name} 失败，停止后续步骤")
                break
                
        except Exception as e:
            print(f"\n❌ {step_name} 出现异常: {e}")
            results.append((step_name, False))
            break
    
    # 总结
    print(f"\n📊 执行结果总结:")
    print("=" * 60)
    for step_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {step_name}: {status}")
    
    all_success = all(success for _, success in results)
    if all_success:
        print(f"\n🎉 所有步骤执行成功！")
        print("现在可以正常运行特征提取了:")
        print("python feature_extraction/extract_features.py --data_root ../MURA-v1.1 --csv_path dataset_csv/mura_abnormality_detection.csv --output_dir processed_data/low_res_features --model_name clip_RN50")
    else:
        print(f"\n⚠️ 部分步骤失败，请检查上述错误信息")

if __name__ == "__main__":
    main()