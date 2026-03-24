# coding=utf-8
"""
身体部位感知的ViLa_MIL模型
仿照ViLa-MIL-main最初版的双尺度并行匹配机制，但根据身体部位和标签动态选择文本提示词
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join as pjoin
from .model_utils import *
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
import warnings

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# 添加父目录到路径以导入body_part_text_matcher
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from body_part_text_matcher import BodyPartTextMatcher

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class BodyPartAwarePromptLearner(nn.Module):
    """身体部位感知的提示词学习器"""
    
    def __init__(self, classnames, clip_model, body_part_matcher):
        super().__init__()
        self.body_part_matcher = body_part_matcher
        n_cls = len(classnames)
        n_ctx = 4  # 减少上下文长度以确保不超过CLIP的最大长度
        ctx_init = ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        # 注册为buffer，这样会自动跟随模型的设备移动
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  
        self.register_buffer("token_suffix", embedding[:, 1:-n_ctx, :]) 

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = "end"
        self.dtype = dtype
        self.clip_model = clip_model

    def forward(self, body_part=None, label=None):
        """
        前向传播，根据身体部位和标签动态生成提示词
        
        参数:
            body_part: 身体部位 (如 'XR_WRIST')
            label: 标签 (0=正常, 1=异常)
        """
        if body_part is not None and label is not None:
            # 动态生成特定的文本提示词
            return self._generate_dynamic_prompts(body_part, label)
        else:
            # 使用默认的通用提示词
            return self._generate_default_prompts()
    
    def _generate_dynamic_prompts(self, body_part, label):
        """生成动态提示词"""
        # 获取特定的文本提示词
        text_prompts = self.body_part_matcher.get_text_prompts_for_sample(body_part, label)
        
        # 为两个分辨率分别编码提示词
        all_prompts = []
        tokenized_prompts_list = []
        
        for prompt_text in text_prompts:
            if prompt_text:  # 确保提示词不为空
                tokenized = clip.tokenize(prompt_text)
                # 确保tokenized张量在正确的设备上
                tokenized = tokenized.to(next(self.clip_model.parameters()).device)
                
                # 确保tokenized也不超过最大长度
                max_length = 77
                if tokenized.shape[1] > max_length:
                    tokenized = tokenized[:, :max_length]
                
                tokenized_prompts_list.append(tokenized)
                
                with torch.no_grad():
                    embedding = self.clip_model.token_embedding(tokenized).type(self.dtype)
                
                # 添加可学习的上下文
                ctx = self.ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
                
                # 构建完整的提示词，确保不超过CLIP的最大长度
                prefix = embedding[:, :1, :]  # [CLS] token
                suffix = embedding[:, 1:-1, :]  # 原始文本（去掉EOS）
                eot = embedding[:, -1:, :]  # [EOS] token
                
                # 计算可用的序列长度（CLIP最大长度是77）
                max_length = 77
                available_length = max_length - 2  # 减去CLS和EOS
                ctx_length = ctx.shape[1]
                suffix_length = suffix.shape[1]
                
                # 如果上下文+原文本太长，截断suffix
                if ctx_length + suffix_length > available_length:
                    suffix = suffix[:, :available_length - ctx_length, :]
                
                # 拼接：[CLS] + ctx + suffix + [EOS]
                full_prompt = torch.cat([prefix, ctx, suffix, eot], dim=1)
                
                # 确保最终长度不超过77
                if full_prompt.shape[1] > max_length:
                    full_prompt = full_prompt[:, :max_length, :]
                all_prompts.append(full_prompt)
        
        if all_prompts:
            prompts = torch.cat(all_prompts, dim=0)
            tokenized_prompts = torch.cat(tokenized_prompts_list, dim=0)
            return prompts, tokenized_prompts
        else:
            # 回退到默认提示词
            return self._generate_default_prompts()
    
    def _generate_default_prompts(self):
        """生成默认提示词"""
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        else:
            # 其他位置的处理逻辑...
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        return prompts, self.tokenized_prompts


class ViLa_MIL_BodyPartAware_Model(nn.Module):
    """身体部位感知的ViLa_MIL模型"""
    
    def __init__(self, config, num_classes=2, 
                 body_part_prompt_path='text_prompt/mura_body_part_text_prompt.csv',
                 general_prompt_path='text_prompt/mura_two_scale_text_prompt.csv'):
        super(ViLa_MIL_BodyPartAware_Model, self).__init__()
        
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1
        
        # 注意力机制
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        # 初始化身体部位文本匹配器
        print("正在初始化身体部位文本匹配器...", flush=True)
        self.body_part_matcher = BodyPartTextMatcher(body_part_prompt_path, general_prompt_path)
        print("身体部位文本匹配器初始化完成", flush=True)
        
        # 加载CLIP模型 - 完全按照原始代码的方式
        print("正在加载CLIP模型...", flush=True)
        clip_model, _ = clip.load("RN50", device="cpu")
        print("CLIP模型加载完成", flush=True)
        self.clip_model = clip_model.float()
        print("CLIP模型转换为float完成", flush=True)
        
        # 身体部位感知的提示词学习器
        print("正在获取默认类别名称...", flush=True)
        default_classnames = self.body_part_matcher.get_all_text_prompts_for_class(num_classes)
        print(f"默认类别名称: {default_classnames[:2]}...", flush=True)
        print("正在初始化提示词学习器...", flush=True)
        self.prompt_learner = BodyPartAwarePromptLearner(default_classnames, self.clip_model, self.body_part_matcher)
        print("提示词学习器初始化完成", flush=True)
        print("正在初始化文本编码器...", flush=True)
        self.text_encoder = TextEncoder(self.clip_model)
        print("文本编码器初始化完成", flush=True)

        # 标准化和交叉注意力
        self.norm = nn.LayerNorm(config.input_size)
        self.cross_attention_1 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)

        # 可学习的图像中心
        self.learnable_image_center = nn.Parameter(torch.Tensor(*[config.prototype_number, 1, config.input_size]))
        trunc_normal_(self.learnable_image_center, std=.02)

    def forward(self, x_s, coord_s, x_l, coords_l, label, body_part=None):
        """
        前向传播 - 身体部位感知的双尺度匹配
        
        参数:
            x_s: 低分辨率图像特征
            coord_s: 低分辨率坐标
            x_l: 高分辨率图像特征  
            coords_l: 高分辨率坐标
            label: 真实标签
            body_part: 身体部位信息
        """
        # 单文本分支（两类：正常/异常），不使用标签泄漏：始终使用共享E_T与共享CTD
        # 身体部位仅用于默认提示词模板选择（可选），不在推理时注入真实标签
        prompts, tokenized_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)  # D，形状约为[num_classes, dim]

        # 处理低分辨率图像特征
        M = x_s.float()
        compents, _ = self.cross_attention_1(self.learnable_image_center, M, M) 
        compents = self.norm(compents + self.learnable_image_center)

        # 处理高分辨率图像特征
        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)

        # 注意力池化 - 低分辨率
        H = compents.squeeze().float()
        A_V = self.attention_V(H)  
        A_U = self.attention_U(H)  
        A = self.attention_weights(A_V * A_U) 
        A = torch.transpose(A, 1, 0)  
        A = F.softmax(A, dim=1)  
        image_features_low = torch.mm(A, H)  

        # 注意力池化 - 高分辨率
        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)  
        A_U_high = self.attention_U(H_high)  
        A_high = self.attention_weights(A_V_high * A_U_high) 
        A_high = torch.transpose(A_high, 1, 0)  
        A_high = F.softmax(A_high, dim=1)  
        image_features_high = torch.mm(A_high, H_high)  

        # 共享CTD在低/高上下文下各调用一次，得到 D'_l、D'_h
        D = text_features  # [num_classes, dim]
        D = D.to(M.device)
        # 低倍上下文增强
        image_context_low = torch.cat((compents.squeeze(), M), dim=0)
        D_l_ctx, _ = self.cross_attention_2(D.unsqueeze(1), image_context_low, image_context_low)
        D_l = D_l_ctx.squeeze() + D
        # 高倍上下文增强
        image_context_high = torch.cat((compents_high.squeeze(), M_high), dim=0)
        D_h_ctx, _ = self.cross_attention_2(D.unsqueeze(1), image_context_high, image_context_high)
        D_h = D_h_ctx.squeeze() + D

        # 相似度得分
        device = image_features_low.device
        logits_low = image_features_low @ D_l.T.to(device)
        logits_high = image_features_high @ D_h.T.to(device)

        # 按类别在两尺度间softmax得到 α_l / α_h（图1要求）
        stacked = torch.stack([logits_low, logits_high], dim=0)  # [2, 1, C]
        alpha = torch.softmax(stacked, dim=0)
        logits = alpha[0] * logits_low + alpha[1] * logits_high

        # 计算损失和预测
        loss = self.loss_ce(logits, label)
        
        # 调试信息：检查logits是否有数值问题
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"警告: logits包含NaN或Inf值")
            print(f"logits范围: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
        
        # 添加训练/测试模式的调试信息
        if not self.training:
            print(f"测试模式 - logits范围: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
            print(f"测试模式 - body_part: {body_part}, label: {label.item() if hasattr(label, 'item') else label}")
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return logits, Y_prob, Y_hat, loss


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    import math
    import warnings
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)