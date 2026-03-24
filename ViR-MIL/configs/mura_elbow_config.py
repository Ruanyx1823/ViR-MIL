"""
MURA肘部(XR_ELBOW)数据集配置文件
包含肘部特定的实验配置参数
"""

import ml_collections
from configs.mura_config import get_config as get_base_config

def get_config():
    """获取MURA肘部数据集配置"""
    config = get_base_config()
    
    # 肘部特定配置
    config.dataset.body_part = 'XR_ELBOW'
    config.split.split_dir = 'task_mura_abnormality_detection_100/XR_ELBOW'
    config.text_prompt.path = 'text_prompt/body_parts/mura_xr_elbow_text_prompt.csv'
    
    # 肘部特定模型参数
    config.model.prototype_number = 12  # 肘部图像特征复杂度适中
    
    # 肘部特定训练参数
    config.train.max_epochs = 100
    config.train.lr = 1e-4
    
    # 肘部特定实验代码
    config.results.exp_code = 'mura_elbow_experiment'
    
    return config