"""
MURA肩部(XR_SHOULDER)数据集配置文件
包含肩部特定的实验配置参数
"""

import ml_collections
from configs.mura_config import get_config as get_base_config

def get_config():
    """获取MURA肩部数据集配置"""
    config = get_base_config()
    
    # 肩部特定配置
    config.dataset.body_part = 'XR_SHOULDER'
    config.split.split_dir = 'task_mura_abnormality_detection_100/XR_SHOULDER'
    config.text_prompt.path = 'text_prompt/body_parts/mura_xr_shoulder_text_prompt.csv'
    
    # 肩部特定模型参数
    config.model.prototype_number = 20  # 肩部图像结构复杂，需要更多原型
    config.model.hidden_size = 224  # 增加隐藏层大小以捕获更复杂的特征
    
    # 肩部特定训练参数
    config.train.max_epochs = 150  # 肩部数据量大，结构复杂，需要更多轮次
    config.train.lr = 5e-5  # 较小的学习率以稳定训练
    config.train.weight_decay = 2e-5  # 增加权重衰减以防止过拟合
    
    # 肩部特定实验代码
    config.results.exp_code = 'mura_shoulder_experiment'
    
    return config