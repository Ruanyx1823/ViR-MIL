"""
MURA配置文件工具模块
提供加载和处理配置文件的功能
"""

import os
import argparse
import importlib
import ml_collections
import json

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA配置文件工具')
    parser.add_argument('--config_name', type=str, default='mura_config',
                        help='配置文件名称（不含.py扩展名）')
    parser.add_argument('--body_part', type=str, default=None,
                        choices=[None, 'XR_WRIST', 'XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER'],
                        help='身体部位，如果指定则加载该部位特定配置')
    parser.add_argument('--training_mode', type=str, default='standard',
                        choices=['fast', 'standard', 'full', 'weighted', 'ensemble'],
                        help='训练模式')
    parser.add_argument('--output_path', type=str, default=None,
                        help='配置输出路径（JSON格式）')
    parser.add_argument('--merge_args', action='store_true', default=False,
                        help='是否将命令行参数合并到配置中')
    return parser.parse_args()

def load_config(config_name='mura_config'):
    """
    加载配置文件
    
    参数:
        config_name: 配置文件名称（不含.py扩展名）
    
    返回:
        config: 配置对象
    """
    try:
        config_module = importlib.import_module(f'configs.{config_name}')
        config = config_module.get_config()
        return config
    except ImportError as e:
        print(f"错误: 无法导入配置文件 '{config_name}': {e}")
        return None
    except AttributeError as e:
        print(f"错误: 配置文件 '{config_name}' 中没有 get_config 函数: {e}")
        return None

def load_body_part_config(body_part):
    """
    加载身体部位特定配置
    
    参数:
        body_part: 身体部位名称
    
    返回:
        config: 配置对象
    """
    config_name = f'mura_{body_part.lower()}_config'
    return load_config(config_name)

def load_training_config(training_mode):
    """
    加载训练模式特定配置
    
    参数:
        training_mode: 训练模式
    
    返回:
        config: 配置对象
    """
    try:
        config_module = importlib.import_module('configs.mura_training_configs')
        
        if training_mode == 'fast':
            return config_module.get_fast_training_config()
        elif training_mode == 'standard':
            return config_module.get_standard_training_config()
        elif training_mode == 'full':
            return config_module.get_full_training_config()
        elif training_mode == 'weighted':
            return config_module.get_weighted_sampling_config()
        elif training_mode == 'ensemble':
            return config_module.get_ensemble_training_config()
        else:
            print(f"警告: 未知的训练模式 '{training_mode}'，使用标准配置")
            return config_module.get_standard_training_config()
    except ImportError as e:
        print(f"错误: 无法导入训练配置: {e}")
        return None
    except AttributeError as e:
        print(f"错误: 训练配置中没有所需的函数: {e}")
        return None

def merge_configs(base_config, override_config):
    """
    合并配置
    
    参数:
        base_config: 基础配置
        override_config: 覆盖配置
    
    返回:
        merged_config: 合并后的配置
    """
    if base_config is None:
        return override_config
    if override_config is None:
        return base_config
    
    merged_config = ml_collections.ConfigDict(base_config)
    
    # 递归合并配置
    def _merge(base, override):
        for k, v in override.items():
            if k in base and isinstance(v, ml_collections.ConfigDict) and isinstance(base[k], ml_collections.ConfigDict):
                _merge(base[k], v)
            else:
                base[k] = v
    
    _merge(merged_config, override_config)
    return merged_config

def config_to_args(config):
    """
    将配置转换为命令行参数
    
    参数:
        config: 配置对象
    
    返回:
        args: 参数列表
    """
    args = []
    
    # 数据集参数
    if hasattr(config, 'dataset'):
        if hasattr(config.dataset, 'csv_path'):
            args.extend(['--csv_path', config.dataset.csv_path])
        if hasattr(config.dataset, 'data_root_dir'):
            args.extend(['--data_root_dir', config.dataset.data_root_dir])
        if hasattr(config.dataset, 'data_folder_s'):
            args.extend(['--data_folder_s', config.dataset.data_folder_s])
        if hasattr(config.dataset, 'data_folder_l'):
            args.extend(['--data_folder_l', config.dataset.data_folder_l])
        if hasattr(config.dataset, 'task'):
            args.extend(['--task', config.dataset.task])
        if hasattr(config.dataset, 'body_part') and config.dataset.body_part is not None:
            args.extend(['--body_part', config.dataset.body_part])
    
    # 分割参数
    if hasattr(config, 'split'):
        if hasattr(config.split, 'split_dir'):
            args.extend(['--split_dir', config.split.split_dir])
        if hasattr(config.split, 'label_frac'):
            args.extend(['--label_frac', str(config.split.label_frac)])
        if hasattr(config.split, 'k'):
            args.extend(['--k', str(config.split.k)])
    
    # 模型参数
    if hasattr(config, 'model'):
        if hasattr(config.model, 'model_type'):
            args.extend(['--model_type', config.model.model_type])
        if hasattr(config.model, 'mode'):
            args.extend(['--mode', config.model.mode])
        if hasattr(config.model, 'prototype_number'):
            args.extend(['--prototype_number', str(config.model.prototype_number)])
        if hasattr(config.model, 'drop_out') and config.model.drop_out:
            args.append('--drop_out')
    
    # 文本提示词参数
    if hasattr(config, 'text_prompt') and hasattr(config.text_prompt, 'path'):
        args.extend(['--text_prompt_path', config.text_prompt.path])
    elif hasattr(config, 'text_prompt') and hasattr(config.text_prompt, 'general_path'):
        args.extend(['--text_prompt_path', config.text_prompt.general_path])
    
    # 训练参数
    if hasattr(config, 'train'):
        if hasattr(config.train, 'max_epochs'):
            args.extend(['--max_epochs', str(config.train.max_epochs)])
        if hasattr(config.train, 'lr'):
            args.extend(['--lr', str(config.train.lr)])
        if hasattr(config.train, 'weight_decay'):
            args.extend(['--reg', str(config.train.weight_decay)])
        if hasattr(config.train, 'optimizer'):
            args.extend(['--opt', config.train.optimizer])
        if hasattr(config.train, 'loss'):
            args.extend(['--bag_loss', config.train.loss])
        if hasattr(config.train, 'seed'):
            args.extend(['--seed', str(config.train.seed)])
        if hasattr(config.train, 'early_stopping') and config.train.early_stopping:
            args.append('--early_stopping')
        if hasattr(config.train, 'weighted_sample') and config.train.weighted_sample:
            args.append('--weighted_sample')
    
    # 结果参数
    if hasattr(config, 'results'):
        if hasattr(config.results, 'results_dir'):
            args.extend(['--results_dir', config.results.results_dir])
        if hasattr(config.results, 'exp_code'):
            args.extend(['--exp_code', config.results.exp_code])
        if hasattr(config.results, 'log_data') and config.results.log_data:
            args.append('--log_data')
    
    return args

def save_config_to_json(config, output_path):
    """
    将配置保存为JSON文件
    
    参数:
        config: 配置对象
        output_path: 输出路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将ConfigDict转换为普通字典
    def _to_dict(config_dict):
        result = {}
        for k, v in config_dict.items():
            if isinstance(v, ml_collections.ConfigDict):
                result[k] = _to_dict(v)
            else:
                result[k] = v
        return result
    
    config_dict = _to_dict(config)
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {output_path}")

def main():
    """主函数"""
    args = parse_args()
    
    # 加载基础配置
    config = load_config(args.config_name)
    
    # 如果指定了身体部位，加载身体部位特定配置
    if args.body_part:
        body_part_config = load_body_part_config(args.body_part.lower())
        if body_part_config:
            config = merge_configs(config, body_part_config)
    
    # 加载训练模式特定配置
    training_config = load_training_config(args.training_mode)
    if training_config:
        config = merge_configs(config, training_config)
    
    # 如果指定了输出路径，保存配置
    if args.output_path:
        save_config_to_json(config, args.output_path)
    
    # 将配置转换为命令行参数
    cmd_args = config_to_args(config)
    print("\n命令行参数:")
    print(' '.join(cmd_args))

if __name__ == "__main__":
    main()