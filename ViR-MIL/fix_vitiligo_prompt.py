import os
import sys

def fix_vitiligo_prompt(input_path='text_prompt/vitiligo_two_scale_text_prompt.csv', output_path=None):
    """
    修复白癜风文本提示文件中的格式问题
    
    参数:
        input_path (str): 输入文件路径
        output_path (str, optional): 输出文件路径，如果为None则覆盖输入文件
    
    返回:
        bool: 是否成功修复
    """
    print(f"开始修复文本提示文件: {input_path}")
    
    # 确保文件存在
    if not os.path.exists(input_path):
        print(f"错误: 未找到文本提示文件 {input_path}")
        return False
    
    # 如果未指定输出路径，则覆盖原文件
    if output_path is None:
        output_path = input_path
    
    try:
        # 读取原始文件
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复格式问题
        lines = content.strip().split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # 跳过空行
            if not line.strip():
                continue
                
            # 跳过无关内容
            if '询问' in line or 'ChatGPT' in line:
                continue
            
            # 修复引号不匹配问题
            if not line.endswith('"') and line.startswith('"'):
                line += '"'
            
            # 确保每行都有引号
            if not line.startswith('"'):
                line = '"' + line
            if not line.endswith('"'):
                line = line + '"'
            
            fixed_lines.append(line)
        
        # 确保有四行提示（两个类别，每个类别两个分辨率）
        if len(fixed_lines) < 4:
            print(f"警告: 修复后的文件只有 {len(fixed_lines)} 行，应该有 4 行")
            
            # 如果缺少行，尝试复制现有行并修改
            while len(fixed_lines) < 4:
                if len(fixed_lines) == 0:
                    # 如果没有任何行，创建默认提示
                    if len(fixed_lines) == 0:
                        fixed_lines.append('"A WSI of stable vitiligo at low resolution with visually descriptive characteristics of well-demarcated depigmented patches, regular borders, smooth edges, and uniform color."')
                    if len(fixed_lines) == 1:
                        fixed_lines.append('"A WSI of developing vitiligo at low resolution with visually descriptive characteristics of irregular depigmented patches, undefined borders, variable size, and a mixed color of white and light beige."')
                    if len(fixed_lines) == 2:
                        fixed_lines.append('"A WSI of stable vitiligo at high resolution with visually descriptive characteristics of clear depigmented areas, melanocyte disappearance, minimal perifollicular repigmentation, and low inflammatory activity."')
                    if len(fixed_lines) == 3:
                        fixed_lines.append('"A WSI of developing vitiligo at high resolution with visually descriptive characteristics of melanocyte loss, active edge inflammation, increased lymphocytic infiltration, and irregular repigmentation patterns."')
                else:
                    # 复制最后一行并修改
                    last_line = fixed_lines[-1]
                    if "low resolution" in last_line:
                        new_line = last_line.replace("low resolution", "high resolution")
                    elif "high resolution" in last_line:
                        new_line = last_line.replace("high resolution", "low resolution")
                    elif "stable" in last_line:
                        new_line = last_line.replace("stable", "developing")
                    elif "developing" in last_line:
                        new_line = last_line.replace("developing", "stable")
                    else:
                        new_line = last_line
                    fixed_lines.append(new_line)
        
        # 确保文本提示顺序正确：stable低分辨率，developing低分辨率，stable高分辨率，developing高分辨率
        ordered_lines = []
        stable_low = None
        developing_low = None
        stable_high = None
        developing_high = None
        
        for line in fixed_lines:
            if "stable" in line.lower() and "low resolution" in line.lower():
                stable_low = line
            elif "developing" in line.lower() and "low resolution" in line.lower():
                developing_low = line
            elif "stable" in line.lower() and "high resolution" in line.lower():
                stable_high = line
            elif "developing" in line.lower() and "high resolution" in line.lower():
                developing_high = line
        
        # 如果找不到某些类别，使用默认值
        if not stable_low:
            stable_low = '"A WSI of stable vitiligo at low resolution with visually descriptive characteristics of well-demarcated depigmented patches, regular borders, smooth edges, and uniform color."'
        if not developing_low:
            developing_low = '"A WSI of developing vitiligo at low resolution with visually descriptive characteristics of irregular depigmented patches, undefined borders, variable size, and a mixed color of white and light beige."'
        if not stable_high:
            stable_high = '"A WSI of stable vitiligo at high resolution with visually descriptive characteristics of clear depigmented areas, melanocyte disappearance, minimal perifollicular repigmentation, and low inflammatory activity."'
        if not developing_high:
            developing_high = '"A WSI of developing vitiligo at high resolution with visually descriptive characteristics of melanocyte loss, active edge inflammation, increased lymphocytic infiltration, and irregular repigmentation patterns."'
        
        # 按正确顺序添加
        ordered_lines = [stable_low, developing_low, stable_high, developing_high]
        
        # 写入修复后的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ordered_lines))
        
        print(f"文本提示文件已修复并保存至: {output_path}")
        print("修复后的内容:")
        for i, line in enumerate(ordered_lines):
            print(f"{i+1}: {line}")
        
        return True
    
    except Exception as e:
        print(f"修复文件时出错: {str(e)}")
        return False

def ensure_text_prompt_dir():
    """
    确保文本提示目录存在
    """
    os.makedirs('text_prompt', exist_ok=True)

def check_and_create_prompt_file():
    """
    检查并创建文本提示文件（如果不存在）
    """
    prompt_path = 'text_prompt/vitiligo_two_scale_text_prompt.csv'
    
    if not os.path.exists(prompt_path):
        print(f"未找到文本提示文件: {prompt_path}")
        print("创建默认文本提示文件...")
        
        # 创建默认文本提示
        default_prompts = [
            '"A WSI of stable vitiligo at low resolution with visually descriptive characteristics of well-demarcated depigmented patches, regular borders, smooth edges, and uniform color."',
            '"A WSI of developing vitiligo at low resolution with visually descriptive characteristics of irregular depigmented patches, undefined borders, variable size, and a mixed color of white and light beige."',
            '"A WSI of stable vitiligo at high resolution with visually descriptive characteristics of clear depigmented areas, melanocyte disappearance, minimal perifollicular repigmentation, and low inflammatory activity."',
            '"A WSI of developing vitiligo at high resolution with visually descriptive characteristics of melanocyte loss, active edge inflammation, increased lymphocytic infiltration, and irregular repigmentation patterns."'
        ]
        
        # 确保目录存在
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        
        # 写入默认提示
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_prompts))
        
        print(f"已创建默认文本提示文件: {prompt_path}")
        return True
    
    return False

if __name__ == "__main__":
    # 确保文本提示目录存在
    ensure_text_prompt_dir()
    
    # 检查并创建文本提示文件（如果不存在）
    created = check_and_create_prompt_file()
    
    # 修复文本提示文件
    if fix_vitiligo_prompt():
        print("文本提示文件修复成功")
    else:
        print("文本提示文件修复失败")