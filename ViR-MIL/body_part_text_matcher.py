"""
身体部位感知的文本提示词匹配器
仿照ViLa-MIL-main最初版的双尺度并行匹配机制，但根据身体部位和标签动态选择文本提示词
"""

import pandas as pd
import numpy as np
import torch
import os
from typing import Dict, List, Tuple, Optional

class BodyPartTextMatcher:
    """身体部位感知的文本提示词匹配器"""
    
    def __init__(self, 
                 body_part_prompt_path: str = 'text_prompt/mura_body_part_text_prompt.csv',
                 general_prompt_path: str = 'text_prompt/mura_two_scale_text_prompt.csv'):
        """
        初始化文本提示词匹配器
        
        参数:
            body_part_prompt_path: 身体部位特定文本提示词文件路径
            general_prompt_path: 通用文本提示词文件路径
        """
        self.body_part_prompt_path = body_part_prompt_path
        self.general_prompt_path = general_prompt_path
        
        # 加载文本提示词
        print("正在加载身体部位特定提示词...", flush=True)
        self.body_part_prompts = self._load_body_part_prompts()
        print("正在加载通用提示词...", flush=True)
        self.general_prompts = self._load_general_prompts()
        
        # 身体部位映射
        self.body_parts = ['XR_WRIST', 'XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 
                          'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER']
        
        print(f"加载了 {len(self.body_part_prompts)} 个身体部位的特定提示词")
        print(f"加载了 {len(self.general_prompts)} 个通用提示词")
    
    def _load_body_part_prompts(self) -> Dict[str, Dict[str, str]]:
        """
        加载身体部位特定的文本提示词
        
        返回:
            body_part_prompts: 按身体部位和类型组织的文本提示词字典
        """
        if not os.path.exists(self.body_part_prompt_path):
            print(f"警告: 身体部位提示词文件不存在 {self.body_part_prompt_path}")
            return {}
        
        # 读取CSV文件（假设每行是一个提示词）
        with open(self.body_part_prompt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().strip('"') for line in f.readlines() if line.strip()]
        
        body_part_prompts = {}
        
        # 解析提示词，按照固定顺序：negative_low, positive_low, negative_high, positive_high
        # 每个身体部位有4个提示词
        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                # 从提示词中提取身体部位
                body_part = self._extract_body_part_from_prompt(lines[i])
                if body_part:
                    body_part_prompts[body_part] = {
                        'negative_low': lines[i],      # 正常低分辨率
                        'positive_low': lines[i + 1],  # 异常低分辨率
                        'negative_high': lines[i + 2], # 正常高分辨率
                        'positive_high': lines[i + 3]  # 异常高分辨率
                    }
        
        return body_part_prompts
    
    def _extract_body_part_from_prompt(self, prompt: str) -> Optional[str]:
        """
        从文本提示词中提取身体部位信息
        
        参数:
            prompt: 文本提示词
            
        返回:
            body_part: 身体部位名称
        """
        prompt_lower = prompt.lower()
        
        if "wrist" in prompt_lower:
            return "XR_WRIST"
        elif "elbow" in prompt_lower:
            return "XR_ELBOW"
        elif "finger" in prompt_lower:
            return "XR_FINGER"
        elif "forearm" in prompt_lower:
            return "XR_FOREARM"
        elif "hand" in prompt_lower:
            return "XR_HAND"
        elif "humerus" in prompt_lower:
            return "XR_HUMERUS"
        elif "shoulder" in prompt_lower:
            return "XR_SHOULDER"
        
        return None
    
    def _load_general_prompts(self) -> Dict[str, str]:
        """
        加载通用文本提示词
        
        返回:
            general_prompts: 通用文本提示词字典
        """
        if not os.path.exists(self.general_prompt_path):
            print(f"警告: 通用提示词文件不存在 {self.general_prompt_path}")
            return {}
        
        # 读取CSV文件
        with open(self.general_prompt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().strip('"') for line in f.readlines() if line.strip()]
        
        # 按照固定顺序组织通用提示词
        general_prompts = {}
        if len(lines) >= 4:
            general_prompts = {
                'abnormal_low': lines[0],   # 异常低分辨率
                'normal_low': lines[1],     # 正常低分辨率
                'abnormal_high': lines[2],  # 异常高分辨率
                'normal_high': lines[3]     # 正常高分辨率
            }
        
        return general_prompts
    
    def get_text_prompts_for_sample(self, body_part: str, label: int) -> List[str]:
        """
        为特定样本获取文本提示词（模仿原版的双尺度匹配）
        
        参数:
            body_part: 身体部位 (如 'XR_WRIST')
            label: 标签 (0=正常, 1=异常)
            
        返回:
            text_prompts: 4个文本提示词列表 [low_res_text, high_res_text]
                         实际返回对应标签的低分辨率和高分辨率提示词
        """
        # 确定提示词类型
        if label == 0:  # 正常
            low_key = 'negative_low'
            high_key = 'negative_high'
            general_low_key = 'normal_low'
            general_high_key = 'normal_high'
        else:  # 异常
            low_key = 'positive_low'
            high_key = 'positive_high'
            general_low_key = 'abnormal_low'
            general_high_key = 'abnormal_high'
        
        # 优先使用身体部位特定的提示词
        if body_part in self.body_part_prompts:
            specific_prompts = self.body_part_prompts[body_part]
            low_text = specific_prompts.get(low_key, "")
            high_text = specific_prompts.get(high_key, "")
        else:
            # 回退到通用提示词
            low_text = self.general_prompts.get(general_low_key, "")
            high_text = self.general_prompts.get(general_high_key, "")
        
        return [low_text, high_text]
    
    def get_all_text_prompts_for_class(self, num_classes: int = 2) -> List[str]:
        """
        获取所有类别的文本提示词（用于模型初始化）
        
        参数:
            num_classes: 类别数量
            
        返回:
            all_prompts: 所有文本提示词列表
                        格式: [class0_low, class1_low, class0_high, class1_high]
        """
        # 使用通用提示词作为默认
        all_prompts = []
        
        if num_classes == 2:
            # 二分类：正常和异常
            all_prompts = [
                self.general_prompts.get('normal_low', ''),     # 正常低分辨率
                self.general_prompts.get('abnormal_low', ''),   # 异常低分辨率  
                self.general_prompts.get('normal_high', ''),    # 正常高分辨率
                self.general_prompts.get('abnormal_high', '')   # 异常高分辨率
            ]
        
        return all_prompts
    
    def create_dynamic_text_features(self, body_parts: List[str], labels: List[int], 
                                   text_encoder, clip_model) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为一个批次的样本创建动态文本特征
        
        参数:
            body_parts: 身体部位列表
            labels: 标签列表
            text_encoder: 文本编码器
            clip_model: CLIP模型
            
        返回:
            text_features_low: 低分辨率文本特征
            text_features_high: 高分辨率文本特征
        """
        batch_size = len(body_parts)
        text_features_low_list = []
        text_features_high_list = []
        
        for i in range(batch_size):
            # 获取当前样本的文本提示词
            prompts = self.get_text_prompts_for_sample(body_parts[i], labels[i])
            
            # 编码文本提示词
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype)
            
            # 这里需要进一步处理...（简化版本）
            # 实际实现中需要完整的文本编码流程
            
        return text_features_low_list, text_features_high_list


def test_body_part_text_matcher():
    """测试身体部位文本匹配器"""
    print("=== 测试身体部位文本匹配器 ===")
    
    # 创建匹配器
    matcher = BodyPartTextMatcher()
    
    # 测试样本
    test_cases = [
        ('XR_WRIST', 0),  # 腕部正常
        ('XR_WRIST', 1),  # 腕部异常
        ('XR_ELBOW', 0),  # 肘部正常
        ('XR_ELBOW', 1),  # 肘部异常
        ('UNKNOWN', 0),   # 未知部位（应该回退到通用提示词）
    ]
    
    for body_part, label in test_cases:
        prompts = matcher.get_text_prompts_for_sample(body_part, label)
        label_name = "正常" if label == 0 else "异常"
        print(f"\n{body_part} - {label_name}:")
        print(f"  低分辨率: {prompts[0][:100]}...")
        print(f"  高分辨率: {prompts[1][:100]}...")
    
    # 测试通用提示词
    all_prompts = matcher.get_all_text_prompts_for_class(2)
    print(f"\n通用提示词数量: {len(all_prompts)}")


if __name__ == "__main__":
    test_body_part_text_matcher()