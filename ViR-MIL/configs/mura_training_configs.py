"""
MURA训练参数配置文件
包含不同训练策略的配置参数
"""

import ml_collections
from configs.mura_config import get_config as get_base_config

def get_fast_training_config():
    """获取快速训练配置（用于调试或快速实验）"""
    config = get_base_config()
    
    # 快速训练参数
    config.train.max_epochs = 50
    config.train.lr = 2e-4
    config.train.patience = 10
    config.results.exp_code = 'mura_fast_training'
    
    return config

def get_standard_training_config():
    """获取标准训练配置（用于一般实验）"""
    config = get_base_config()
    
    # 标准训练参数
    config.train.max_epochs = 100
    config.train.lr = 1e-4
    config.train.patience = 20
    config.results.exp_code = 'mura_standard_training'
    
    return config

def get_full_training_config():
    """获取完整训练配置（用于最终模型）"""
    config = get_base_config()
    
    # 完整训练参数
    config.train.max_epochs = 200
    config.train.lr = 5e-5
    config.train.patience = 30
    config.train.weight_decay = 2e-5
    config.results.exp_code = 'mura_full_training'
    
    return config

def get_lr_sweep_configs():
    """获取学习率扫描配置列表"""
    lr_values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    configs = []
    
    for lr in lr_values:
        config = get_base_config()
        config.train.lr = lr
        config.results.exp_code = f'mura_lr_{str(lr).replace("-", "_")}'
        configs.append(config)
    
    return configs

def get_prototype_sweep_configs():
    """获取原型数量扫描配置列表"""
    prototype_values = [8, 12, 16, 24, 32]
    configs = []
    
    for proto_num in prototype_values:
        config = get_base_config()
        config.model.prototype_number = proto_num
        config.results.exp_code = f'mura_proto_{proto_num}'
        configs.append(config)
    
    return configs

def get_weighted_sampling_config():
    """获取加权采样配置（用于不平衡数据）"""
    config = get_base_config()
    
    # 加权采样参数
    config.train.weighted_sample = True
    config.results.exp_code = 'mura_weighted_sampling'
    
    return config

def get_ensemble_training_config():
    """获取集成学习配置"""
    config = get_base_config()
    
    # 集成学习参数
    config.train.ensemble = True
    config.train.ensemble_size = 5
    config.train.ensemble_seeds = [42, 43, 44, 45, 46]
    config.results.exp_code = 'mura_ensemble'
    
    return config