"""
修复文本提示文件中的格式问题
"""

import os

def fix_text_prompt_file(input_path, output_path=None):
    """
    修复文本提示文件中的格式问题
    
    参数:
        input_path (str): 输入文件路径
        output_path (str, optional): 输出文件路径，如果为None则覆盖输入文件
    """
    if output_path is None:
        output_path = input_path
    
    # 读取原始文件
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复格式问题
    lines = content.strip().split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if i == 3:  # 第四行有问题
            if not line.endswith('"'):
                line += '"'
        
        # 跳过空行和无关内容
        if line and not line.startswith('询问'):
            fixed_lines.append(line)
    
    # 写入修复后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"文本提示文件已修复并保存至 {output_path}")

if __name__ == "__main__":
    # 修复文本提示文件
    fix_text_prompt_file('text_prompt/vitiligo_two_scale_text_prompt.csv')
    print("文本提示文件修复完成")