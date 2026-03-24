"""
MURA数据集配置文件
包含通用实验配置参数
"""

import ml_collections
import os

def get_config():
    """获取MURA数据集通用配置"""
    config = ml_collections.ConfigDict()

    # 数据集配置
    config.dataset = ml_collections.ConfigDict()
    config.dataset.name = 'MURA'
    config.dataset.task = 'task_mura_abnormality_detection'
    config.dataset.csv_path = 'dataset_csv/mura_abnormality_detection.csv'
    config.dataset.data_root_dir = 'processed_data'
    config.dataset.data_folder_s = 'low_res_features'
    config.dataset.data_folder_l = 'high_res_features'
    config.dataset.label_dict = {'normal': 0, 'abnormal': 1}
    config.dataset.n_classes = 2
    config.dataset.body_parts = ['XR_WRIST', 'XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER']
    
    # 分割配置
    config.split = ml_collections.ConfigDict()
    config.split.split_dir = 'task_mura_abnormality_detection_100/stratified_body_part'
    config.split.label_frac = 1.0
    config.split.k = 5
    
    # 模型配置
    config.model = ml_collections.ConfigDict()
    config.model.model_type = 'ViLa_MIL'
    config.model.mode = 'transformer'
    config.model.input_size = 1024
    config.model.hidden_size = 192
    config.model.prototype_number = 16
    config.model.drop_out = True
    
    # 文本提示词配置
    config.text_prompt = ml_collections.ConfigDict()
    config.text_prompt.general_path = 'text_prompt/mura_two_scale_text_prompt.csv'
    config.text_prompt.body_part_dir = 'text_prompt/body_parts'
    
    # 训练配置
    config.train = ml_collections.ConfigDict()
    config.train.max_epochs = 100
    config.train.batch_size = 1  # MIL通常使用batch_size=1
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-5
    config.train.early_stopping = True
    config.train.patience = 20
    config.train.optimizer = 'adam'
    config.train.loss = 'ce'
    config.train.seed = 42
    config.train.weighted_sample = False
    
    # 结果配置
    config.results = ml_collections.ConfigDict()
    config.results.results_dir = 'results/mura_experiment'
    config.results.log_data = True
    
    return config