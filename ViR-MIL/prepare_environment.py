import os
import sys
import shutil
import subprocess
from tqdm import tqdm

def check_python_packages():
    """
    检查必要的Python包是否已安装
    
    返回:
        tuple: (是否全部已安装, 缺少的包列表)
    """
    print("检查Python包...")
    
    required_packages = [
        'torch',
        'torchvision',
        'h5py',
        'pandas',
        'numpy',
        'tqdm',
        'Pillow',
        'clip',
        'openslide-python',
        'scikit-learn',
        'scipy',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        return False, missing_packages
    
    return True, []

def install_missing_packages(packages):
    """
    安装缺少的Python包
    
    参数:
        packages (list): 要安装的包列表
    
    返回:
        bool: 是否全部安装成功
    """
    print(f"安装缺少的Python包: {', '.join(packages)}")
    
    all_success = True
    
    for package in packages:
        print(f"安装 {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  - {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"  - {package} 安装失败")
            all_success = False
    
    return all_success

def create_directory_structure():
    """
    创建必要的目录结构
    
    返回:
        bool: 是否成功创建
    """
    print("创建目录结构...")
    
    directories = [
        'dataset_csv',
        'splits',
        'text_prompt',
        'processed_data',
        'processed_data/patches_256',
        'processed_data/low_res_features',
        'processed_data/high_res_features',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  - 创建目录: {directory}")
    
    return True

def check_main_py_modification():
    """
    检查main.py是否已修改
    
    返回:
        bool: 是否已修改
    """
    print("检查main.py是否已修改...")
    
    if not os.path.exists('main.py'):
        print("  - 未找到main.py文件")
        return False
    
    with open('main.py', 'r') as f:
        content = f.read()
    
    if 'task_vitiligo_subtyping' in content:
        print("  - main.py已包含白癜风任务")
        return True
    
    print("  - main.py未包含白癜风任务")
    return False

def modify_main_py():
    """
    修改main.py文件，添加白癜风任务
    
    返回:
        bool: 是否成功修改
    """
    print("修改main.py文件...")
    
    if not os.path.exists('main.py'):
        print("  - 未找到main.py文件")
        return False
    
    # 读取main.py文件
    with open('main.py', 'r') as f:
        content = f.readlines()
    
    # 读取main_patch.py文件
    if not os.path.exists('main_patch.py'):
        print("  - 未找到main_patch.py文件")
        return False
    
    with open('main_patch.py', 'r') as f:
        patch_content = f.read()
    
    # 提取补丁代码
    import re
    patch_code = re.search(r'"""(.*?)"""', patch_content, re.DOTALL)
    if not patch_code:
        print("  - 无法从main_patch.py提取补丁代码")
        return False
    
    patch_code = patch_code.group(1).strip()
    
    # 寻找插入位置
    insert_position = -1
    for i, line in enumerate(content):
        if 'raise NotImplementedError' in line:
            insert_position = i
            break
    
    if insert_position == -1:
        print("  - 无法找到插入位置")
        return False
    
    # 备份原始文件
    shutil.copy('main.py', 'main.py.bak')
    print("  - 已备份原始main.py文件为main.py.bak")
    
    # 插入补丁代码
    modified_content = content[:insert_position]
    modified_content.append(patch_code + '\n')
    modified_content.extend(content[insert_position:])
    
    # 写入修改后的文件
    with open('main.py', 'w') as f:
        f.writelines(modified_content)
    
    print("  - main.py文件已修改")
    return True

def generate_run_script():
    """
    生成运行脚本
    
    返回:
        bool: 是否成功生成
    """
    print("生成运行脚本...")
    
    # 生成Windows批处理文件
    with open('run_vila_mil.bat', 'w') as f:
        f.write('@echo off\n')
        f.write('echo 运行ViLa-MIL模型...\n')
        f.write('python main.py ^\n')
        f.write('--seed 1 ^\n')
        f.write('--drop_out ^\n')
        f.write('--early_stopping ^\n')
        f.write('--lr 1e-4 ^\n')
        f.write('--k 5 ^\n')
        f.write('--label_frac 1 ^\n')
        f.write('--bag_loss ce ^\n')
        f.write('--task "task_vitiligo_subtyping" ^\n')
        f.write('--results_dir "./results" ^\n')
        f.write('--exp_code "vitiligo_exp" ^\n')
        f.write('--model_type ViLa_MIL ^\n')
        f.write('--mode transformer ^\n')
        f.write('--log_data ^\n')
        f.write('--data_root_dir "./processed_data" ^\n')
        f.write('--data_folder_s "low_res_features" ^\n')
        f.write('--data_folder_l "high_res_features" ^\n')
        f.write('--split_dir "task_vitiligo_subtyping_100" ^\n')
        f.write('--text_prompt_path "./text_prompt/vitiligo_two_scale_text_prompt.csv" ^\n')
        f.write('--prototype_number 16\n')
        f.write('pause\n')
    
    # 生成Linux/macOS Shell脚本
    with open('run_vila_mil.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('echo "运行ViLa-MIL模型..."\n')
        f.write('python main.py \\\n')
        f.write('--seed 1 \\\n')
        f.write('--drop_out \\\n')
        f.write('--early_stopping \\\n')
        f.write('--lr 1e-4 \\\n')
        f.write('--k 5 \\\n')
        f.write('--label_frac 1 \\\n')
        f.write('--bag_loss ce \\\n')
        f.write('--task "task_vitiligo_subtyping" \\\n')
        f.write('--results_dir "./results" \\\n')
        f.write('--exp_code "vitiligo_exp" \\\n')
        f.write('--model_type ViLa_MIL \\\n')
        f.write('--mode transformer \\\n')
        f.write('--log_data \\\n')
        f.write('--data_root_dir "./processed_data" \\\n')
        f.write('--data_folder_s "low_res_features" \\\n')
        f.write('--data_folder_l "high_res_features" \\\n')
        f.write('--split_dir "task_vitiligo_subtyping_100" \\\n')
        f.write('--text_prompt_path "./text_prompt/vitiligo_two_scale_text_prompt.csv" \\\n')
        f.write('--prototype_number 16\n')
    
    # 设置Shell脚本可执行权限
    try:
        os.chmod('run_vila_mil.sh', 0o755)
    except:
        pass
    
    print("  - 已生成运行脚本: run_vila_mil.bat (Windows)")
    print("  - 已生成运行脚本: run_vila_mil.sh (Linux/macOS)")
    
    return True

def prepare_environment():
    """
    准备运行环境
    
    返回:
        bool: 是否成功准备
    """
    steps = [
        ("检查Python包", check_python_packages),
        ("创建目录结构", create_directory_structure),
        ("检查main.py修改", check_main_py_modification),
        ("生成运行脚本", generate_run_script)
    ]
    
    all_success = True
    
    for step_name, step_func in steps:
        print(f"\n=== {step_name} ===")
        
        if step_name == "检查Python包":
            success, missing_packages = step_func()
            if not success:
                print(f"缺少以下Python包: {', '.join(missing_packages)}")
                install_success = install_missing_packages(missing_packages)
                if not install_success:
                    all_success = False
        elif step_name == "检查main.py修改":
            modified = step_func()
            if not modified:
                print("尝试修改main.py...")
                if not modify_main_py():
                    all_success = False
        else:
            if not step_func():
                all_success = False
    
    if all_success:
        print("\n环境准备完成，可以运行ViLa-MIL模型")
        print("\n运行方式:")
        print("  - Windows: 双击运行 run_vila_mil.bat")
        print("  - Linux/macOS: 在终端中执行 ./run_vila_mil.sh")
    else:
        print("\n环境准备失败，请修复上述问题后重试")
    
    return all_success

if __name__ == "__main__":
    prepare_environment()