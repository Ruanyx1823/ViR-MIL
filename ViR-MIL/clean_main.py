import os
import sys
import shutil
import re

def backup_main_py():
    """
    备份原始main.py文件
    
    返回:
        bool: 是否成功备份
    """
    if not os.path.exists('main.py'):
        print("错误: 未找到main.py文件")
        return False
    
    backup_path = 'main.py.original'
    if not os.path.exists(backup_path):
        shutil.copy('main.py', backup_path)
        print(f"已备份原始main.py文件为 {backup_path}")
    else:
        print(f"备份文件 {backup_path} 已存在，跳过备份")
    
    return True

def modify_main_py():
    """
    修改main.py文件，删除其他文本提示的代码，只保留白癜风任务
    
    返回:
        bool: 是否成功修改
    """
    if not os.path.exists('main.py'):
        print("错误: 未找到main.py文件")
        return False
    
    # 读取main.py文件
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 删除TCGA_RCC和TCGA_Lung任务的代码
    modified_content = re.sub(
        r"if args\.task == 'task_tcga_rcc_subtyping':(.*?)elif args\.task == 'task_tcga_lung_subtyping':(.*?)else:",
        "if args.task == 'task_vitiligo_subtyping':\n"
        "    args.n_classes=2\n"
        "    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/vitiligo_subtyping.csv',\n"
        "                                  mode = args.mode,\n"
        "                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),\n"
        "                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),\n"
        "                                  shuffle = False,\n"
        "                                  print_info = True,\n"
        "                                  label_dict = {'Stable': 0, 'Developing': 1},\n"
        "                                  patient_strat= False,\n"
        "                                  ignore=[])\nelse:",
        content,
        flags=re.DOTALL
    )
    
    # 写入修改后的文件
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("已修改main.py文件，删除其他文本提示的代码，只保留白癜风任务")
    return True

def clean_text_prompt_dir():
    """
    清理text_prompt目录，只保留白癜风文本提示
    
    返回:
        bool: 是否成功清理
    """
    text_prompt_dir = 'text_prompt'
    if not os.path.exists(text_prompt_dir):
        os.makedirs(text_prompt_dir)
        print(f"创建目录: {text_prompt_dir}")
    
    # 获取目录中的所有文件
    files = os.listdir(text_prompt_dir)
    
    # 创建备份目录
    backup_dir = os.path.join(text_prompt_dir, 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    
    # 移动非白癜风文本提示文件到备份目录
    for file in files:
        if file != 'vitiligo_two_scale_text_prompt.csv' and file != 'backup':
            src_path = os.path.join(text_prompt_dir, file)
            dst_path = os.path.join(backup_dir, file)
            if os.path.isfile(src_path):
                shutil.move(src_path, dst_path)
                print(f"已将 {file} 移动到备份目录")
    
    # 确保白癜风文本提示文件存在
    vitiligo_prompt_path = os.path.join(text_prompt_dir, 'vitiligo_two_scale_text_prompt.csv')
    if not os.path.exists(vitiligo_prompt_path):
        print(f"警告: 未找到白癜风文本提示文件 {vitiligo_prompt_path}")
        print("尝试创建默认文本提示文件...")
        
        # 创建默认文本提示
        default_prompts = [
            '"A WSI of stable vitiligo at low resolution with visually descriptive characteristics of well-demarcated depigmented patches, regular borders, smooth edges, and uniform color."',
            '"A WSI of developing vitiligo at low resolution with visually descriptive characteristics of irregular depigmented patches, undefined borders, variable size, and a mixed color of white and light beige."',
            '"A WSI of stable vitiligo at high resolution with visually descriptive characteristics of clear depigmented areas, melanocyte disappearance, minimal perifollicular repigmentation, and low inflammatory activity."',
            '"A WSI of developing vitiligo at high resolution with visually descriptive characteristics of melanocyte loss, active edge inflammation, increased lymphocytic infiltration, and irregular repigmentation patterns."'
        ]
        
        # 写入默认提示
        with open(vitiligo_prompt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_prompts))
        
        print(f"已创建默认白癜风文本提示文件: {vitiligo_prompt_path}")
    
    print("已清理text_prompt目录，只保留白癜风文本提示")
    return True

def test_modified_code():
    """
    测试修改后的代码，确保没有错误
    
    返回:
        bool: 是否测试通过
    """
    print("测试修改后的代码...")
    
    # 检查main.py文件
    if not os.path.exists('main.py'):
        print("错误: 未找到main.py文件")
        return False
    
    # 读取main.py文件
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否包含白癜风任务
    if 'task_vitiligo_subtyping' not in content:
        print("错误: main.py文件中未找到白癜风任务")
        return False
    
    # 检查是否删除了其他任务
    if 'task_tcga_rcc_subtyping' in content and 'TCGA_RCC_subtyping.csv' in content:
        print("警告: main.py文件中仍包含TCGA_RCC任务")
    
    if 'task_tcga_lung_subtyping' in content and 'TCGA_Lung_subtyping.csv' in content:
        print("警告: main.py文件中仍包含TCGA_Lung任务")
    
    # 检查白癜风文本提示文件
    vitiligo_prompt_path = 'text_prompt/vitiligo_two_scale_text_prompt.csv'
    if not os.path.exists(vitiligo_prompt_path):
        print(f"错误: 未找到白癜风文本提示文件 {vitiligo_prompt_path}")
        return False
    
    print("修改后的代码测试通过")
    return True

def clean_all():
    """
    执行所有清理步骤
    
    返回:
        bool: 是否全部成功
    """
    steps = [
        ("备份main.py文件", backup_main_py),
        ("修改main.py文件", modify_main_py),
        ("清理text_prompt目录", clean_text_prompt_dir),
        ("测试修改后的代码", test_modified_code)
    ]
    
    all_success = True
    
    for step_name, step_func in steps:
        print(f"\n=== {step_name} ===")
        if not step_func():
            print(f"{step_name} 失败")
            all_success = False
    
    if all_success:
        print("\n所有清理步骤执行成功")
    else:
        print("\n部分清理步骤执行失败，请检查上述错误信息")
    
    return all_success

if __name__ == "__main__":
    print("此脚本将删除main.py中其他文本提示的代码，只保留白癜风任务")
    print("同时清理text_prompt目录，只保留白癜风文本提示文件")
    print("是否继续？(y/n)")
    
    response = input().strip().lower()
    if response == 'y':
        clean_all()
    else:
        print("已取消执行")