import os
import sys
import subprocess
from tqdm import tqdm

def run_command(command, description=None):
    """
    运行命令并显示输出
    
    参数:
        command (str): 要运行的命令
        description (str, optional): 命令描述
    
    返回:
        tuple: (是否成功, 输出)
    """
    if description:
        print(f"\n=== {description} ===")
    
    print(f"运行命令: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        output = []
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            output.append(line)
        
        process.wait()
        
        if process.returncode == 0:
            print(f"命令执行成功")
            return True, '\n'.join(output)
        else:
            print(f"命令执行失败，返回码: {process.returncode}")
            return False, '\n'.join(output)
    
    except Exception as e:
        print(f"命令执行出错: {str(e)}")
        return False, str(e)

def run_all():
    """
    运行所有步骤
    
    返回:
        bool: 是否全部成功
    """
    steps = [
        ("数据处理", "python vitiligo_data_processing.py"),
        ("文本提示修复", "python fix_vitiligo_prompt.py"),
        ("数据集分割", "python create_vitiligo_splits.py"),
        ("特征提取", "python vitiligo_feature_extraction.py"),
        ("兼容性验证", "python verify_compatibility.py"),
        ("环境准备", "python prepare_environment.py")
    ]
    
    results = []
    
    for step_name, command in steps:
        print(f"\n\n{'='*50}")
        print(f"步骤: {step_name}")
        print(f"{'='*50}\n")
        
        success, _ = run_command(command, step_name)
        results.append((step_name, success))
        
        if not success:
            print(f"\n{step_name} 失败，是否继续？(y/n)")
            response = input().strip().lower()
            if response != 'y':
                print(f"停止后续步骤")
                break
    
    print("\n\n执行结果汇总:")
    for step_name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{status}: {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n所有步骤执行成功，系统已准备就绪")
        print("\n可以通过以下方式运行ViLa-MIL模型:")
        print("  - Windows: 双击运行 run_vila_mil.bat")
        print("  - Linux/macOS: 在终端中执行 ./run_vila_mil.sh")
    else:
        print("\n部分步骤执行失败，请修复上述问题后重试")
    
    return all_success

if __name__ == "__main__":
    print("此脚本将依次执行所有步骤，包括数据处理、特征提取等")
    print("整个过程可能需要较长时间，请耐心等待")
    print("是否继续？(y/n)")
    
    response = input().strip().lower()
    if response == 'y':
        run_all()
    else:
        print("已取消执行")