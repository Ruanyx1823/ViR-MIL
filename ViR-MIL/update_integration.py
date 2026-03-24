import os
import sys
import shutil
import re
from tqdm import tqdm

def update_prepare_environment():
    """
    更新prepare_environment.py脚本，反映代码修改
    
    返回:
        bool: 是否成功更新
    """
    file_path = 'prepare_environment.py'
    if not os.path.exists(file_path):
        print(f"错误: 未找到 {file_path} 文件")
        return False
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新modify_main_py函数
    modified_content = re.sub(
        r"def modify_main_py\(\):(.*?)return True",
        """def modify_main_py():
    \"\"\"
    修改main.py文件，添加白癜风任务
    
    返回:
        bool: 是否成功修改
    \"\"\"
    print("修改main.py文件...")
    
    if not os.path.exists('main.py'):
        print("  - 未找到main.py文件")
        return False
    
    # 读取main.py文件
    with open('main.py', 'r') as f:
        content = f.readlines()
    
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
    
    # 插入白癜风任务代码
    vitiligo_code = '''elif args.task == 'task_vitiligo_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/vitiligo_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'Stable': 0, 'Developing': 1},
                                  patient_strat= False,
                                  ignore=[])
'''
    
    # 插入代码
    modified_content = content[:insert_position]
    modified_content.append(vitiligo_code)
    modified_content.extend(content[insert_position:])
    
    # 写入修改后的文件
    with open('main.py', 'w') as f:
        f.writelines(modified_content)
    
    print("  - main.py文件已修改，添加了白癜风任务")
    return True""",
        content,
        flags=re.DOTALL
    )
    
    # 写入修改后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"已更新 {file_path}")
    return True

def update_test_integration():
    """
    更新test_integration.py脚本，反映代码修改
    
    返回:
        bool: 是否成功更新
    """
    file_path = 'test_integration.py'
    if not os.path.exists(file_path):
        print(f"错误: 未找到 {file_path} 文件")
        return False
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加clean_main测试
    modified_content = re.sub(
        r"def test_full_pipeline\(\):(.*?)steps = \[",
        """def test_clean_main():
    \"\"\"
    测试清理main.py脚本
    
    返回:
        bool: 是否成功
    \"\"\"
    return run_command("python clean_main.py", "测试清理main.py脚本")[0]

def test_full_pipeline():
    \"\"\"
    测试完整流程
    
    返回:
        bool: 是否成功
    \"\"\"
    steps = [""",
        content,
        flags=re.DOTALL
    )
    
    # 更新测试步骤
    modified_content = re.sub(
        r"steps = \[(.*?)\]",
        """steps = [
        ("数据处理", test_data_processing),
        ("文本提示修复", test_fix_prompt),
        ("清理main.py", test_clean_main),
        ("数据集分割", test_create_splits),
        ("特征提取测试", test_feature_extraction),
        ("兼容性验证", test_verify_compatibility),
        ("环境准备", test_prepare_environment)
    ]""",
        modified_content,
        flags=re.DOTALL
    )
    
    # 写入修改后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"已更新 {file_path}")
    return True

def update_run_all():
    """
    更新run_all.py脚本，反映代码修改
    
    返回:
        bool: 是否成功更新
    """
    file_path = 'run_all.py'
    if not os.path.exists(file_path):
        print(f"错误: 未找到 {file_path} 文件")
        return False
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新步骤
    modified_content = re.sub(
        r"steps = \[(.*?)\]",
        """steps = [
        ("数据处理", "python vitiligo_data_processing.py"),
        ("文本提示修复", "python fix_vitiligo_prompt.py"),
        ("清理main.py", "python clean_main.py"),
        ("数据集分割", "python create_vitiligo_splits.py"),
        ("特征提取", "python vitiligo_feature_extraction.py"),
        ("兼容性验证", "python verify_compatibility.py"),
        ("环境准备", "python prepare_environment.py")
    ]""",
        content,
        flags=re.DOTALL
    )
    
    # 写入修改后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"已更新 {file_path}")
    return True

def update_readme():
    """
    更新README.md文件，反映代码修改
    
    返回:
        bool: 是否成功更新
    """
    file_path = 'README.md'
    if not os.path.exists(file_path):
        print(f"错误: 未找到 {file_path} 文件")
        return False
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新项目结构
    modified_content = re.sub(
        r"## 项目结构(.*?)## 数据结构",
        """## 项目结构

- `vitiligo_data_processing.py`: 数据处理脚本，实现JPG图像模拟WSI接口和补丁提取
- `vitiligo_feature_extraction.py`: 特征提取脚本，从图像补丁中提取特征
- `fix_vitiligo_prompt.py`: 文本提示文件修复脚本
- `create_vitiligo_splits.py`: 数据集分割脚本
- `clean_main.py`: 清理main.py文件，删除其他文本提示的代码
- `verify_compatibility.py`: 兼容性验证脚本
- `prepare_environment.py`: 环境准备脚本
- `test_integration.py`: 整体流程测试脚本
- `main_patch.py`: 主程序补丁文件

## 数据结构""",
        content,
        flags=re.DOTALL
    )
    
    # 更新使用步骤
    modified_content = re.sub(
        r"## 使用步骤(.*?)## 快速测试",
        """## 使用步骤

### 1. 数据处理

运行数据处理脚本，生成数据集CSV文件并提取图像补丁：

```bash
python vitiligo_data_processing.py
```

### 2. 修复文本提示文件

运行文本提示文件修复脚本：

```bash
python fix_vitiligo_prompt.py
```

### 3. 清理main.py文件

运行清理脚本，删除其他文本提示的代码，只保留白癜风任务：

```bash
python clean_main.py
```

### 4. 创建数据集分割

运行数据集分割脚本：

```bash
python create_vitiligo_splits.py
```

### 5. 特征提取

运行特征提取脚本，从图像补丁中提取特征：

```bash
python vitiligo_feature_extraction.py
```

### 6. 验证兼容性

运行兼容性验证脚本：

```bash
python verify_compatibility.py
```

### 7. 准备运行环境

运行环境准备脚本：

```bash
python prepare_environment.py
```

### 8. 运行模型

运行ViLa-MIL模型：

- Windows: 双击运行 `run_vila_mil.bat`
- Linux/macOS: 在终端中执行 `./run_vila_mil.sh`

## 快速测试""",
        modified_content,
        flags=re.DOTALL
    )
    
    # 写入修改后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"已更新 {file_path}")
    return True

def update_all_scripts():
    """
    更新所有集成脚本
    
    返回:
        bool: 是否全部成功
    """
    steps = [
        ("更新prepare_environment.py", update_prepare_environment),
        ("更新test_integration.py", update_test_integration),
        ("更新run_all.py", update_run_all),
        ("更新README.md", update_readme)
    ]
    
    all_success = True
    
    for step_name, step_func in steps:
        print(f"\n=== {step_name} ===")
        if not step_func():
            print(f"{step_name} 失败")
            all_success = False
    
    if all_success:
        print("\n所有集成脚本更新成功")
    else:
        print("\n部分集成脚本更新失败，请检查上述错误信息")
    
    return all_success

if __name__ == "__main__":
    print("此脚本将更新集成脚本，反映代码修改")
    print("是否继续？(y/n)")
    
    response = input().strip().lower()
    if response == 'y':
        update_all_scripts()
    else:
        print("已取消执行")