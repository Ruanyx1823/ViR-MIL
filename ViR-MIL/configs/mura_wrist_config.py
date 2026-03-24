"""
MURA腕部(XR_WRIST)数据集配置文件
包含腕部特定的实验配置参数
"""

import ml_collections
from configs.mura_config import get_config as get_base_config

def get_config():
    """获取MURA腕部数据集配置"""
    config = get_base_config()
    
    # 腕部特定配置
    config.dataset.body_part = 'XR_WRIST'
    config.split.split_dir = 'task_mura_abnormality_detection_100/XR_WRIST'
    config.text_prompt.path = 'text_prompt/body_parts/mura_xr_wrist_text_prompt.csv'
    
    # 腕部特定模型参数
    config.model.prototype_number = 16  # 腕部图像特征可能需要更多原型
    
    # 腕部特定训练参数
    config.train.max_epochs = 120  # 腕部数据较多，可能需要更多轮次
    config.train.lr = 8e-5  # 腕部特定学习率
    
    # 腕部特定实验代码
    config.results.exp_code = 'mura_wrist_experiment'
    
    return config