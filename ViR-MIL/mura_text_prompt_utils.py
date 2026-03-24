"""
MURA数据集文本提示词处理工具
提供加载、处理和生成文本提示词的功能
"""

import os
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA文本提示词处理工具')
    parser.add_argument('--general_prompt_path', type=str, default='text_prompt/mura_two_scale_text_prompt.csv',
                        help='通用文本提示词路径')
    parser.add_argument('--body_part_prompt_path', type=str, default='text_prompt/mura_body_part_text_prompt.csv',
                        help='身体部位特定文本提示词路径')
    parser.add_argument('--output_path', type=str, default='text_prompt/mura_combined_text_prompt.csv',
                        help='输出文本提示词路径')
    parser.add_argument('--body_part', type=str, default=None,
                        help='指定身体部位（如果不指定，则使用通用提示词）')
    parser.add_argument('--mode', type=str, default='combine',
                        choices=['combine', 'general', 'body_part'],
                        help='处理模式')
    return parser.parse_args()

def load_text_prompts(file_path):
    """
    加载文本提示词
    
    参数:
        file_path: 文本提示词文件路径
    
    返回:
        prompts: 文本提示词列表
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    return prompts

def parse_body_part_prompts(prompts):
    """
    解析身体部位特定的文本提示词
    
    参数:
        prompts: 文本提示词列表
    
    返回:
        body_part_prompts: 按身体部位和类型组织的文本提示词字典
    """
    body_part_prompts = defaultdict(dict)
    
    for prompt in prompts:
        # 提取身体部位
        if "wrist X-ray" in prompt.lower():
            body_part = "XR_WRIST"
        elif "elbow X-ray" in prompt.lower():
            body_part = "XR_ELBOW"
        elif "finger X-ray" in prompt.lower():
            body_part = "XR_FINGER"
        elif "forearm X-ray" in prompt.lower():
            body_part = "XR_FOREARM"
        elif "hand X-ray" in prompt.lower():
            body_part = "XR_HAND"
        elif "humerus X-ray" in prompt.lower():
            body_part = "XR_HUMERUS"
        elif "shoulder X-ray" in prompt.lower():
            body_part = "XR_SHOULDER"
        else:
            continue
        
        # 提取分辨率和异常状态
        if "with abnormality at low resolution" in prompt.lower():
            key = "abnormal_low"
        elif "without abnormality at low resolution" in prompt.lower():
            key = "normal_low"
        elif "with abnormality at high resolution" in prompt.lower():
            key = "abnormal_high"
        elif "without abnormality at high resolution" in prompt.lower():
            key = "normal_high"
        else:
            continue
        
        # 存储提示词
        body_part_prompts[body_part][key] = prompt
    
    return body_part_prompts

def parse_general_prompts(prompts):
    """
    解析通用文本提示词
    
    参数:
        prompts: 文本提示词列表
    
    返回:
        general_prompts: 按类型组织的通用文本提示词字典
    """
    general_prompts = {}
    
    for i, prompt in enumerate(prompts):
        if i == 0:
            general_prompts["abnormal_low"] = prompt
        elif i == 1:
            general_prompts["normal_low"] = prompt
        elif i == 2:
            general_prompts["abnormal_high"] = prompt
        elif i == 3:
            general_prompts["normal_high"] = prompt
    
    return general_prompts

def get_body_part_prompts(body_part_prompts, body_part=None):
    """
    获取指定身体部位的文本提示词
    
    参数:
        body_part_prompts: 按身体部位和类型组织的文本提示词字典
        body_part: 指定的身体部位
    
    返回:
        prompts: 指定身体部位的文本提示词字典
    """
    if body_part is None or body_part not in body_part_prompts:
        # 如果未指定身体部位或指定的身体部位不存在，返回空字典
        return {}
    
    return body_part_prompts[body_part]

def combine_prompts(general_prompts, body_part_prompts, body_part=None):
    """
    合并通用和身体部位特定的文本提示词
    
    参数:
        general_prompts: 通用文本提示词字典
        body_part_prompts: 按身体部位和类型组织的文本提示词字典
        body_part: 指定的身体部位
    
    返回:
        combined_prompts: 合并后的文本提示词列表
    """
    if body_part is not None and body_part in body_part_prompts:
        # 如果指定了身体部位并且存在对应的提示词，使用身体部位特定的提示词
        specific_prompts = body_part_prompts[body_part]
        combined_prompts = [
            specific_prompts.get("abnormal_low", general_prompts.get("abnormal_low", "")),
            specific_prompts.get("normal_low", general_prompts.get("normal_low", "")),
            specific_prompts.get("abnormal_high", general_prompts.get("abnormal_high", "")),
            specific_prompts.get("normal_high", general_prompts.get("normal_high", ""))
        ]
    else:
        # 否则使用通用提示词
        combined_prompts = [
            general_prompts.get("abnormal_low", ""),
            general_prompts.get("normal_low", ""),
            general_prompts.get("abnormal_high", ""),
            general_prompts.get("normal_high", "")
        ]
    
    return combined_prompts

def save_prompts(prompts, output_path):
    """
    保存文本提示词
    
    参数:
        prompts: 文本提示词列表
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    
    print(f"文本提示词已保存到: {output_path}")

def create_all_body_part_prompts(general_prompts, body_part_prompts):
    """
    为所有身体部位创建文本提示词
    
    参数:
        general_prompts: 通用文本提示词字典
        body_part_prompts: 按身体部位和类型组织的文本提示词字典
    """
    body_parts = ["XR_WRIST", "XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER"]
    
    for body_part in body_parts:
        combined_prompts = combine_prompts(general_prompts, body_part_prompts, body_part)
        output_path = f"text_prompt/mura_{body_part.lower()}_text_prompt.csv"
        save_prompts(combined_prompts, output_path)

def main():
    """主函数"""
    args = parse_args()
    
    # 加载通用文本提示词
    general_prompts_list = load_text_prompts(args.general_prompt_path)
    general_prompts = parse_general_prompts(general_prompts_list)
    
    # 加载身体部位特定文本提示词
    body_part_prompts_list = load_text_prompts(args.body_part_prompt_path)
    body_part_prompts = parse_body_part_prompts(body_part_prompts_list)
    
    if args.mode == 'combine':
        # 合并文本提示词
        combined_prompts = combine_prompts(general_prompts, body_part_prompts, args.body_part)
        save_prompts(combined_prompts, args.output_path)
    elif args.mode == 'general':
        # 使用通用文本提示词
        general_prompts_list = [general_prompts.get(key, "") for key in ["abnormal_low", "normal_low", "abnormal_high", "normal_high"]]
        save_prompts(general_prompts_list, args.output_path)
    elif args.mode == 'body_part':
        # 为所有身体部位创建文本提示词
        create_all_body_part_prompts(general_prompts, body_part_prompts)
    
    print("文本提示词处理完成!")

if __name__ == "__main__":
    main()