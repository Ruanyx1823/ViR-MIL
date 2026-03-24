"""
ViLa-MIL模型主程序
用于MURA数据集的骨骼X光片异常检测
"""

from __future__ import print_function
import argparse
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset
import torch
import pandas as pd
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 命令行参数设置
parser = argparse.ArgumentParser(description='ViLa-MIL for MURA Abnormality Detection')
parser.add_argument('--data_root_dir', type=str, default=None, help='数据根目录')
parser.add_argument('--data_folder_s', type=str, default=None, help='低分辨率特征目录')
parser.add_argument('--data_folder_l', type=str, default=None, help='高分辨率特征目录')
parser.add_argument('--max_epochs', type=int, default=200, help='最大训练轮数 (默认: 200)')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率 (默认: 0.001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='训练标签比例 (默认: 1.0)')
parser.add_argument('--seed', type=int, default=1, help='随机种子 (默认: 1)')
parser.add_argument('--k', type=int, default=5, help='交叉验证折数 (默认: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='开始折 (默认: -1, 最后一折)')
parser.add_argument('--k_end', type=int, default=-1, help='结束折 (默认: -1, 第一折)')
parser.add_argument('--results_dir', default='./results', help='结果保存目录 (默认: ./results)')
parser.add_argument('--split_dir', type=str, default=None, help='数据分割目录')
parser.add_argument('--log_data', action='store_true', default=False, help='使用tensorboard记录数据')
parser.add_argument('--testing', action='store_true', default=False, help='测试模式')
parser.add_argument('--early_stopping', action='store_true', default=False, help='启用早停')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam', help='优化器选择')
parser.add_argument('--drop_out', action='store_true', default=False, help='启用dropout (p=0.25)')
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL'], default='ViLa_MIL', help='模型类型')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer', help='模式')
parser.add_argument('--exp_code', type=str, help='实验代码，用于保存结果')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='启用加权采样')
parser.add_argument('--reg', type=float, default=1e-5, help='权重衰减 (默认: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce', help='损失函数')
parser.add_argument('--task', type=str, default='task_mura_abnormality_detection', help='任务名称')
parser.add_argument("--text_prompt", type=str, default=None, help='文本提示词')
parser.add_argument("--text_prompt_path", type=str, default=None, help='文本提示词路径')
parser.add_argument("--prototype_number", type=int, default=16, help='原型数量')
parser.add_argument("--body_part", type=str, default=None, 
                    choices=[None, 'XR_WRIST', 'XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER'],
                    help='身体部位，如果指定则只使用该部位的数据')
parser.add_argument("--csv_path", type=str, default='dataset_csv/mura_abnormality_detection.csv',
                    help='数据集CSV文件路径')
parser.add_argument('--email_notification', action='store_true', default=False, help='发送邮件通知实验完成')
parser.add_argument('--recipient_email', type=str, default=None, help='接收通知的邮箱地址')
parser.add_argument('--smtp_server', type=str, default='smtp.qq.com', help='SMTP服务器地址')
parser.add_argument('--smtp_port', type=int, default=587, help='SMTP服务器端口')
parser.add_argument('--sender_email', type=str, default=None, help='发送通知的邮箱地址')
parser.add_argument('--sender_password', type=str, default=None, help='发送邮箱的授权码')

args = parser.parse_args()

# 加载文本提示词
args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()

def seed_torch(seed=7):
    """设置随机种子"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def send_email_notification(recipient_email, smtp_server, smtp_port, sender_email, sender_password, experiment_name, results_summary):
    """
    发送邮件通知实验完成
    
    参数:
        recipient_email (str): 接收者邮箱
        smtp_server (str): SMTP服务器地址
        smtp_port (int): SMTP服务器端口
        sender_email (str): 发送者邮箱
        sender_password (str): 发送者邮箱授权码
        experiment_name (str): 实验名称
        results_summary (dict): 实验结果摘要
    """
    try:
        # 创建邮件内容
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"实验完成通知: {experiment_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 邮件正文
        body = f"""
        您的MURA实验 {experiment_name} 已经完成!
        
        实验结果摘要:
        - 平均测试AUC: {results_summary.get('test_auc_mean', 'N/A')}
        - 平均测试准确率: {results_summary.get('test_acc_mean', 'N/A')}
        - 平均测试F1分数: {results_summary.get('test_f1_mean', 'N/A')}
        - 实验任务: {results_summary.get('task', 'N/A')}
        - 身体部位: {results_summary.get('body_part', '全部')}
        
        结果保存在: {results_summary.get('results_dir', 'N/A')}
        
        此邮件由自动系统发送，请勿回复。
        """
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 连接到SMTP服务器
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 启用TLS加密
        server.login(sender_email, sender_password)
        
        # 发送邮件
        server.send_message(msg)
        server.quit()
        
        print(f"邮件通知已发送至 {recipient_email}")
        return True
    except Exception as e:
        print(f"发送邮件通知失败: {str(e)}")
        return False

# 设置随机种子
seed_torch(args.seed)

# 设置实验参数
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            'mode': args.mode,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.body_part:
    settings['body_part'] = args.body_part
    print(f'\n使用特定身体部位: {args.body_part}')

print('\n加载数据集')

# 加载MURA数据集
if args.task == 'task_mura_abnormality_detection':
    args.n_classes = 2
    
    # 根据是否指定身体部位选择CSV路径
    csv_path = args.csv_path
    
    # 如果指定了身体部位，过滤数据
    filter_dict = {}
    if args.body_part:
        filter_dict = {'body_part': [args.body_part]}
        # 更新分割目录
        if args.split_dir is None:
            args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)), args.body_part)
        else:
            args.split_dir = os.path.join('splits', args.split_dir, args.body_part)
    
    # 创建数据集
    dataset = Generic_MIL_Dataset(csv_path=csv_path,
                                 mode=args.mode,
                                 data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
                                 data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
                                 shuffle=False,
                                 print_info=True,
                                 label_dict={0: 0, 1: 1},
                                 patient_strat=False,
                                 filter_dict=filter_dict,
                                 ignore=[])
elif args.task == 'task_vitiligo_subtyping':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/vitiligo_subtyping.csv',
                                 mode=args.mode,
                                 data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
                                 data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
                                 shuffle=False,
                                 print_info=True,
                                 label_dict={'Stable': 0, 'Developing': 1},
                                 patient_strat=False,
                                 ignore=[])
else:
    raise NotImplementedError(f"未实现的任务: {args.task}")

# 创建结果目录
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# 设置实验结果目录
if args.body_part:
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + f'_{args.body_part}_s{args.seed}')
else:
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + f'_s{args.seed}')

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# 设置分割目录
if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    # 检查split_dir是否已经包含'splits/'前缀
    if args.split_dir.startswith('splits/'):
        args.split_dir = args.split_dir
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

print('分割目录: ', args.split_dir)
assert os.path.isdir(args.split_dir), f"分割目录不存在: {args.split_dir}"

settings.update({'split_dir': args.split_dir})

# 保存实验设置
with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# 实验设置 ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


def main(args):
    """主函数"""
    # 设置交叉验证折范围
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    # 存储各折结果
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        # 加载数据分割
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) 
        datasets = (train_dataset, val_dataset, test_dataset)
        
        # 训练模型
        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(datasets, i, args)

        # 记录结果
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        
        # 保存结果
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    # 汇总结果
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'test_acc': all_test_acc, 'test_f1': all_test_f1})
    result_df = pd.DataFrame({'metric': ['mean', 'var'],
                             'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
                             'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)],
                             'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
                             })

    # 保存结果
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.results_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    
    return result_df, all_test_auc, all_test_acc, all_test_f1


if __name__ == "__main__":
    results, all_test_auc, all_test_acc, all_test_f1 = main(args)
    print("\n最终结果:")
    print(results)
    
    # 发送邮件通知
    if args.email_notification and args.recipient_email and args.sender_email and args.sender_password:
        # 准备结果摘要
        results_summary = {
            'test_auc_mean': np.mean(all_test_auc) if len(all_test_auc) > 0 else 'N/A',
            'test_acc_mean': np.mean(all_test_acc) if len(all_test_acc) > 0 else 'N/A',
            'test_f1_mean': np.mean(all_test_f1) if len(all_test_f1) > 0 else 'N/A',
            'results_dir': args.results_dir,
            'task': args.task,
            'body_part': args.body_part if args.body_part else '全部'
        }
        
        # 发送邮件通知
        send_email_notification(
            args.recipient_email,
            args.smtp_server,
            args.smtp_port,
            args.sender_email,
            args.sender_password,
            args.exp_code,
            results_summary
        )
    
    print("\n完成!")