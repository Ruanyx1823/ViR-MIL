"""
身体部位感知的ViLa-MIL训练脚本
仿照ViLa-MIL-main最初版的双尺度并行匹配机制，但根据身体部位和标签动态选择文本提示词
"""

from __future__ import print_function
import argparse
import os
import smtplib
import time
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# 防止交互式输入导致nohup挂起
print("=== 程序启动 ===", flush=True)
print(f"启动时间: {datetime.now()}", flush=True)
print("设置stdin重定向...", flush=True)
sys.stdin = open(os.devnull, 'r')
os.environ['PYTHONUNBUFFERED'] = '1'
print("stdin重定向完成", flush=True)
print("开始导入模块...", flush=True)
from utils.file_utils import save_pkl
print("导入file_utils完成", flush=True)
from utils.utils import *
print("导入utils完成", flush=True)
# from utils.core_utils import train  # 使用自定义训练函数
from datasets.dataset_generic_body_part_aware import BodyPartAware_MIL_Dataset
print("导入dataset完成", flush=True)
print("开始导入深度学习库...", flush=True)
import torch
print("导入torch完成", flush=True)
import torch.nn as nn
print("导入torch.nn完成", flush=True)
import torch.optim as optim
print("导入torch.optim完成", flush=True)
import pandas as pd
print("导入pandas完成", flush=True)
import numpy as np
print("导入numpy完成", flush=True)
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
print("导入sklearn完成", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备设置完成: {device}", flush=True)

# 命令行参数设置
parser = argparse.ArgumentParser(description='身体部位感知的ViLa-MIL for MURA Abnormality Detection')

# 数据相关参数
parser.add_argument('--data_root_dir', type=str, default=None, help='数据根目录')
parser.add_argument('--data_folder_s', type=str, default=None, help='低分辨率特征目录')
parser.add_argument('--data_folder_l', type=str, default=None, help='高分辨率特征目录')
parser.add_argument('--csv_path', type=str, default='dataset_csv/mura_abnormality_detection.csv',
                    help='数据集CSV文件路径')

# 训练相关参数
parser.add_argument('--max_epochs', type=int, default=200, help='最大训练轮数 (默认: 200)')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率 (默认: 0.001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='训练标签比例 (默认: 1.0)')
parser.add_argument('--seed', type=int, default=1, help='随机种子 (默认: 1)')
parser.add_argument('--k', type=int, default=5, help='交叉验证折数 (默认: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='开始折 (默认: -1, 最后一折)')
parser.add_argument('--k_end', type=int, default=-1, help='结束折 (默认: -1, 第一折)')
parser.add_argument('--results_dir', default='./results_body_part_aware', help='结果保存目录')
parser.add_argument('--split_dir', type=str, default=None, help='数据分割目录')

# 模型相关参数
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL_BodyPartAware'], 
                    default='ViLa_MIL_BodyPartAware', help='模型类型')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer', help='模式')
parser.add_argument('--drop_out', action='store_true', default=False, help='启用dropout (p=0.25)')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='启用加权采样')
parser.add_argument('--reg', type=float, default=1e-5, help='权重衰减 (默认: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce', help='损失函数')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam', help='优化器选择')

# 文本提示词相关参数
parser.add_argument("--body_part_prompt_path", type=str, 
                    default='text_prompt/mura_body_part_text_prompt.csv',
                    help='身体部位特定文本提示词路径')
parser.add_argument("--general_prompt_path", type=str,
                    default='text_prompt/mura_two_scale_text_prompt.csv', 
                    help='通用文本提示词路径')
parser.add_argument("--prototype_number", type=int, default=16, help='原型数量')

# 实验相关参数
parser.add_argument('--exp_code', type=str, help='实验代码，用于保存结果')
parser.add_argument('--task', type=str, default='task_mura_abnormality_detection', help='任务名称')
parser.add_argument('--testing', action='store_true', default=False, help='测试模式')
parser.add_argument('--early_stopping', action='store_true', default=False, help='启用早停')
parser.add_argument('--log_data', action='store_true', default=False, help='使用tensorboard记录数据')
parser.add_argument('--num_workers', type=int, default=8, help='数据加载器工作进程数 (默认: 8)')

# 身体部位过滤参数（可选）
parser.add_argument("--body_part", type=str, default=None, 
                    choices=[None, 'XR_WRIST', 'XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER'],
                    help='身体部位，如果指定则只使用该部位的数据')

# 邮件通知参数
parser.add_argument('--email_notification', action='store_true', default=False, help='发送邮件通知实验完成')
parser.add_argument('--recipient_email', type=str, default=None, help='接收通知的邮箱地址')
parser.add_argument('--smtp_server', type=str, default='smtp.qq.com', help='SMTP服务器地址')
parser.add_argument('--smtp_port', type=int, default=587, help='SMTP服务器端口')
parser.add_argument('--sender_email', type=str, default=None, help='发送通知的邮箱地址')
parser.add_argument('--sender_password', type=str, default=None, help='发送邮箱的授权码')

print("开始解析命令行参数...", flush=True)
args = parser.parse_args()
print("命令行参数解析完成", flush=True)
print(f"实验代码: {args.exp_code}", flush=True)
print(f"数据根目录: {args.data_root_dir}", flush=True)


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


def calculate_error(Y_hat, Y):
    """计算错误率"""
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error


def send_email_notification(recipient_email, smtp_server, smtp_port, sender_email, 
                          sender_password, experiment_name, results_summary):
    """发送邮件通知实验完成"""
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"身体部位感知实验完成通知: {experiment_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        body = f"""
实验完成通知

实验名称: {experiment_name}
完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

实验结果:
- 测试AUC均值: {results_summary.get('test_auc_mean', 'N/A')}
- 测试准确率均值: {results_summary.get('test_acc_mean', 'N/A')}
- 测试F1均值: {results_summary.get('test_f1_mean', 'N/A')}
- 身体部位: {results_summary.get('body_part', '全部')}
- 结果目录: {results_summary.get('results_dir', 'N/A')}

这是一个自动发送的邮件通知。
        """
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 增加连接超时设置
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
        server.set_debuglevel(1)  # 启用调试模式以获取更多信息
        server.starttls()
        
        # 确保邮箱地址不包含"your_"前缀
        clean_sender_email = sender_email.replace("your_", "")
        
        print(f"尝试登录邮箱: {clean_sender_email}")
        server.login(clean_sender_email, sender_password)
        
        text = msg.as_string()
        server.sendmail(clean_sender_email, recipient_email.replace("your_", ""), text)
        server.quit()
        
        print(f"邮件通知已发送到: {recipient_email.replace('your_', '')}")
        
    except Exception as e:
        print(f"发送邮件通知失败: {e}")
        print("请检查以下可能的问题:")
        print("1. 确保邮箱地址格式正确且不包含'your_'前缀")
        print("2. 确保授权码正确")
        print("3. 确保SMTP服务器和端口设置正确")
        print("4. 确保网络连接正常")


# 设置随机种子
print("开始设置随机种子...", flush=True)
seed_torch(args.seed)
print(f"随机种子设置完成: {args.seed}", flush=True)

# 实验设置
settings = {
    'num_splits': args.k,
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
    'opt': args.opt,
    'body_part_prompt_path': args.body_part_prompt_path,
    'general_prompt_path': args.general_prompt_path
}

print('\n加载数据集', flush=True)
print("检查任务类型...", flush=True)

# 加载MURA数据集
if args.task == 'task_mura_abnormality_detection':
    print("任务类型确认: MURA异常检测", flush=True)
    args.n_classes = 2
    
    # 根据是否指定身体部位选择过滤条件
    filter_dict = {}
    if args.body_part:
        filter_dict = {'body_part': [args.body_part]}
        print(f"过滤身体部位: {args.body_part}")
    
    # 创建身体部位感知的数据集
    print("开始创建数据集对象...", flush=True)
    print(f"CSV路径: {args.csv_path}", flush=True)
    print(f"低分辨率数据目录: {os.path.join(args.data_root_dir, args.data_folder_s)}", flush=True)
    print(f"高分辨率数据目录: {os.path.join(args.data_root_dir, args.data_folder_l)}", flush=True)
    
    dataset = BodyPartAware_MIL_Dataset(
        csv_path=args.csv_path,
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={0: 0, 1: 1},
        patient_strat=False,
        filter_dict=filter_dict,
        ignore=[]
    )
    print("数据集对象创建完成", flush=True)
else:
    raise NotImplementedError(f"未实现的任务: {args.task}")

print("开始创建结果目录...", flush=True)
# 创建结果目录
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
print(f"结果目录创建完成: {args.results_dir}", flush=True)

print("开始设置实验结果目录...", flush=True)
# 设置实验结果目录
if args.body_part:
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + f'_{args.body_part}_s{args.seed}')
else:
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + f'_s{args.seed}')

print(f"实验结果目录路径: {args.results_dir}", flush=True)
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
print("实验结果目录创建完成", flush=True)

print("开始设置分割目录...", flush=True)
# 设置分割目录
if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    if args.split_dir.startswith('splits/'):
        args.split_dir = args.split_dir
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

print('分割目录: ', args.split_dir, flush=True)
print("检查分割目录是否存在...", flush=True)
assert os.path.isdir(args.split_dir), f"分割目录不存在: {args.split_dir}"
print("分割目录验证完成", flush=True)

settings.update({'split_dir': args.split_dir})

print("开始保存实验设置...", flush=True)
# 保存实验设置
try:
    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()
    print("实验设置保存完成", flush=True)
except Exception as e:
    print(f"保存实验设置时出错: {e}", flush=True)
    raise

print("################# 实验设置 ###################", flush=True)
for key, val in settings.items():
    print("{}:  {}".format(key, val), flush=True)
print("实验设置打印完成", flush=True)


def train_body_part_aware(datasets, cur, args):
    """身体部位感知的训练函数"""
    print(f'\n训练第 {cur} 折 - 身体部位感知模式!', flush=True)
    
    print(f"创建写入目录: {args.results_dir}/{cur}", flush=True)
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    print("写入目录创建完成", flush=True)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\n初始化训练/验证/测试分割...', end=' ')
    train_split, val_split, test_split = datasets
    print('完成!')
    print(f"训练样本: {len(train_split)}")
    print(f"验证样本: {len(val_split)}")
    print(f"测试样本: {len(test_split)}")

    # 使用DataLoader以提升GPU利用率
    print('创建DataLoader以提升数据加载吞吐...', flush=True)
    train_loader = get_split_loader(train_split, training=True, testing=False, weighted=args.weighted_sample, mode=args.mode, num_workers=args.num_workers)
    val_loader = get_split_loader(val_split, training=False, testing=False, weighted=False, mode=args.mode, num_workers=args.num_workers)
    test_loader = get_split_loader(test_split, training=False, testing=False, weighted=False, mode=args.mode, num_workers=args.num_workers)

    print('\n初始化损失函数...', end=' ', flush=True)
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        from utils.loss_utils import FocalLoss
        loss_fn = FocalLoss().cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('完成!')
    
    print('\n初始化身体部位感知模型...', end=' ', flush=True)
    print("\n开始导入ml_collections...", flush=True)
    import ml_collections
    print("ml_collections导入完成", flush=True)
    print("开始导入ViLa_MIL_BodyPartAware_Model...", flush=True)
    from models.model_ViLa_MIL_body_part_aware import ViLa_MIL_BodyPartAware_Model
    print("ViLa_MIL_BodyPartAware_Model导入完成", flush=True)
    
    print("创建配置对象...", flush=True)
    config = ml_collections.ConfigDict()
    config.input_size = 1024
    config.hidden_size = 192
    config.prototype_number = args.prototype_number
    print("配置对象创建完成", flush=True)
    
    print("准备模型参数字典...", flush=True)
    model_dict = {
        'config': config, 
        'num_classes': args.n_classes,
        'body_part_prompt_path': args.body_part_prompt_path,
        'general_prompt_path': args.general_prompt_path
    }
    print("模型参数字典准备完成", flush=True)
    print("开始创建模型实例...", flush=True)
    model = ViLa_MIL_BodyPartAware_Model(**model_dict)
    print('模型创建完成!', flush=True)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)

    print('\n初始化优化器...', end=' ')
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    print('完成!')

    if args.early_stopping:
        from utils.core_utils import EarlyStopping
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
    else:
        early_stopping = None

    # 开始训练
    best_model_saved = False
    for epoch in range(args.max_epochs):
        # 训练一个epoch的身体部位感知逻辑
        train_loop_body_part_aware(epoch, model, train_loader, optimizer, args.n_classes, 
                                  writer, loss_fn)
        
        # 验证
        stop = validate_body_part_aware(cur, epoch, model, val_loader, args.n_classes, 
                                      early_stopping, writer, loss_fn, args.results_dir)
        
        # 检查是否保存了最佳模型
        if args.early_stopping and early_stopping:
            checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
            if os.path.exists(checkpoint_path):
                best_model_saved = True
        
        if stop:
            break
    
    # 如果没有通过早停保存模型，手动保存最后的模型
    if args.early_stopping and not best_model_saved:
        checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
        print(f"早停未保存模型，手动保存最终模型到: {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)

    # 测试
    if args.early_stopping:
        checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            print(f"加载最佳模型: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"警告: 最佳模型文件不存在: {checkpoint_path}")
    
    results_dict, test_error, test_auc, df = summary_body_part_aware(
        model, test_loader, args.n_classes)
    
    print('测试错误率: {:.4f}, 测试AUC: {:.4f}'.format(test_error, test_auc))
    
    if writer:
        writer.close()
    
    return results_dict, test_auc, 1-test_error, df


def train_loop_body_part_aware(epoch, model, loader, optimizer, n_classes, writer, loss_fn):
    """身体部位感知的训练循环"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    train_loss = 0.
    train_error = 0.
    
    print(f'\n训练 Epoch: {epoch}')
    
    for batch_idx, batch_data in enumerate(loader):
        # 解包身体部位感知的数据
        if len(batch_data) == 6:  # 包含身体部位信息
            data_s, coords_s, data_l, coords_l, label, body_part = batch_data
        else:  # 回退到原始格式
            data_s, coords_s, data_l, coords_l, label = batch_data
            body_part = None
        
        data_s, data_l = data_s.to(device), data_l.to(device)
        
        # 确保标签是tensor格式，并保持正确的维度
        if not isinstance(label, torch.Tensor):
            label = torch.tensor([label])  # 确保是1D张量
        elif label.dim() == 0:  # 如果是0维张量，添加维度
            label = label.unsqueeze(0)
        label = label.to(device)
        
        # 前向传播（传入身体部位信息）
        logits, Y_prob, Y_hat, loss = model(data_s, coords_s, data_l, coords_l, label, body_part)
        
        loss_value = loss.item()
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            label_value = label[0].item() if isinstance(label, torch.Tensor) and label.numel() > 0 else label
            current_body_part = body_part[0] if isinstance(body_part, (list, tuple)) else body_part
            print(f'batch {batch_idx}, loss: {loss_value:.4f}, '
                  f'label: {label_value}, body_part: {current_body_part}')
    
    train_loss /= len(loader)
    train_error /= len(loader)
    
    print(f'Epoch: {epoch}, 训练损失: {train_loss:.4f}, 训练错误率: {train_error:.4f}')
    
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate_body_part_aware(cur, epoch, model, loader, n_classes, early_stopping, writer, loss_fn, results_dir):
    """身体部位感知的验证"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    val_loss = 0.
    val_error = 0.
    
    # 修复：使用列表而不是固定大小数组，以便处理不同大小的批次
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            # 解包身体部位感知的数据
            if len(batch_data) == 6:
                data_s, coords_s, data_l, coords_l, label, body_part = batch_data
            else:
                data_s, coords_s, data_l, coords_l, label = batch_data
                body_part = None
            
            data_s, data_l = data_s.to(device), data_l.to(device)
            
            # 确保标签是tensor格式，并保持正确的维度 (验证)
            if not isinstance(label, torch.Tensor):
                label = torch.tensor([label])  # 确保是1D张量
            elif label.dim() == 0:  # 如果是0维张量，添加维度
                label = label.unsqueeze(0)
            label = label.to(device)
            
            logits, Y_prob, Y_hat, loss = model(data_s, coords_s, data_l, coords_l, label, body_part)
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            
            # 修复：收集所有预测和标签
            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(label[0].item() if isinstance(label, torch.Tensor) and label.numel() > 0 else label)
    
    val_error /= len(loader)
    val_loss /= len(loader)
    
    # 修复：将列表转换为适当的格式来计算AUC
    all_probs = np.vstack(all_probs) if len(all_probs) > 0 else np.array([])
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) < 2:
        # 如果只有一个类别，无法计算AUC，设为0.5
        print("警告: 验证集只包含一个类别，无法计算AUC")
        auc = 0.5
    elif n_classes == 2 and all_probs.shape[1] == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    print(f'验证 Epoch: {epoch}, 验证损失: {val_loss:.4f}, 验证错误率: {val_error:.4f}, 验证AUC: {auc:.4f}')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/auc', auc, epoch)
    
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    
    return False


def summary_body_part_aware(model, loader, n_classes):
    """身体部位感知的测试总结"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    test_loss = 0.
    test_error = 0.
    
    all_probs = []
    all_labels = []
    all_preds = []
    
    # 修复：从loader.dataset获取slide_data
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            # 解包身体部位感知的数据
            if len(batch_data) == 6:
                data_s, coords_s, data_l, coords_l, label, body_part = batch_data
            else:
                data_s, coords_s, data_l, coords_l, label = batch_data
                body_part = None
            
            data_s, data_l = data_s.to(device), data_l.to(device)
            
            # 确保标签是tensor格式，并保持正确的维度 (测试)
            if not isinstance(label, torch.Tensor):
                label = torch.tensor([label])  # 确保是1D张量
            elif label.dim() == 0:  # 如果是0维张量，添加维度
                label = label.unsqueeze(0)
            label = label.to(device)
            
            slide_id = slide_ids.iloc[batch_idx]
            
            logits, Y_prob, Y_hat, loss = model(data_s, coords_s, data_l, coords_l, label, body_part)
            
            test_loss += loss.item()
            error = calculate_error(Y_hat, label)
            test_error += error
            
            probs = Y_prob.cpu().numpy()
            all_probs.append(probs)  # 修复：添加整个概率数组
            all_labels.append(label[0].item() if isinstance(label, torch.Tensor) and label.numel() > 0 else label)
            all_preds.append(Y_hat.item())
            
            patient_results[slide_id] = {
                'slide_id': slide_id,
                'Y': label[0].item() if isinstance(label, torch.Tensor) and label.numel() > 0 else label,
                'Y_hat': Y_hat.item(),
                'Y_prob': probs,
                'body_part': body_part
            }
    
    test_error /= len(loader)
    test_loss /= len(loader)
    
    # 修复：将列表转换为适当的格式来计算AUC
    all_probs = np.vstack(all_probs) if len(all_probs) > 0 else np.array([])
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) < 2:
        # 如果只有一个类别，无法计算AUC，设为0.5
        print("警告: 测试集只包含一个类别，无法计算AUC")
        auc = 0.5
    elif n_classes == 2 and all_probs.shape[1] == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    results_dict = {
        'slide_id': list(patient_results.keys()),
        'Y': all_labels,
        'Y_hat': all_preds,
        'Y_prob': all_probs,
        'auc': auc,
        'test_error': test_error
    }
    
    df = pd.DataFrame({
        'slide_id': list(patient_results.keys()),
        'Y': all_labels,
        'Y_hat': all_preds,
        'body_part': [patient_results[sid]['body_part'] for sid in patient_results.keys()]
    })
    
    return results_dict, test_error, auc, df


def print_dataset_stats(dataset, name):
    """打印数据集统计信息"""
    if dataset is None:
        print(f"{name}数据集为空")
        return
    
    print(f"\n=== {name}数据集统计 ===")
    print(f"总样本数: {len(dataset)}")
    
    # 统计标签分布
    labels = []
    body_parts = []
    
    for i in range(len(dataset)):
        try:
            # 尝试获取样本信息
            sample = dataset[i]
            if len(sample) >= 6:  # 包含body_part信息
                _, _, _, _, label, body_part = sample
                labels.append(label)
                body_parts.append(body_part)
            else:
                print(f"样本 {i} 格式异常，跳过统计")
        except Exception as e:
            print(f"样本 {i} 加载失败: {e}")
            continue
    
    if labels:
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("标签分布:")
        for label, count in zip(unique_labels, counts):
            label_name = "正常" if label == 0 else "异常"
            print(f"  {label_name} (标签 {label}): {count} 样本 ({count/len(labels)*100:.1f}%)")
        
        if body_parts:
            unique_parts, part_counts = np.unique([bp for bp in body_parts if bp is not None], return_counts=True)
            print("身体部位分布:")
            for part, count in zip(unique_parts, part_counts):
                print(f"  {part}: {count} 样本 ({count/len(body_parts)*100:.1f}%)")
    else:
        print("无法获取样本统计信息")


def main(args):
    """主函数"""
    print("进入main函数...", flush=True)
    print(f"k_start: {args.k_start}, k_end: {args.k_end}, k: {args.k}", flush=True)
    
    # 设置交叉验证折范围
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    
    print(f"计算得到的折数范围: start={start}, end={end}", flush=True)

    # 存储各折结果
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    
    folds = np.arange(start, end)
    print(f"生成的折数数组: {folds}", flush=True)
    print(f"折数数组长度: {len(folds)}", flush=True)
    
    if len(folds) == 0:
        print("错误: 没有有效的折数进行训练!", flush=True)
        print(f"请检查参数设置: k={args.k}, k_start={args.k_start}, k_end={args.k_end}", flush=True)
        return None, [], [], []
    
    for i in folds:
        seed_torch(args.seed)
        print(f"\n{'='*50}", flush=True)
        print(f"开始第 {i} 折训练", flush=True)
        print(f"{'='*50}", flush=True)
        
        # 加载数据分割
        print(f"正在加载第{i}折的数据分割...", flush=True)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        print(f"第{i}折数据分割加载完成", flush=True) 
        
        # 打印数据集统计信息
        print_dataset_stats(train_dataset, f"第{i}折训练")
        print_dataset_stats(val_dataset, f"第{i}折验证")
        print_dataset_stats(test_dataset, f"第{i}折测试")
        
        datasets = (train_dataset, val_dataset, test_dataset)
        
        # 使用身体部位感知的训练函数
        results, test_auc, test_acc, test_df = train_body_part_aware(datasets, i, args)

        # 记录结果
        all_test_auc.append(test_auc)
        all_test_acc.append(test_acc)
        
        # 计算F1分数、特异性和敏感性
        from sklearn.metrics import f1_score
        test_f1 = f1_score(test_df['Y'], test_df['Y_hat'], average='weighted')
        all_test_f1.append(test_f1)
        
        # 计算特异性和敏感性（仅适用于二分类）
        test_sensitivity = -1
        test_specificity = -1
        if args.n_classes == 2:
            cm = confusion_matrix(test_df['Y'], test_df['Y_hat'])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 敏感性 = 真阳性率
                test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性 = 真阴性率
        
        print(f'第{i}折测试结果:')
        print(f'  准确率: {test_acc:.4f}')
        print(f'  AUC: {test_auc:.4f}')
        print(f'  F1分数: {test_f1:.4f}')
        print(f'  敏感性: {test_sensitivity:.4f}')
        print(f'  特异性: {test_specificity:.4f}')
        
        # 保存结果
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    # 汇总结果
    final_df = pd.DataFrame({
        'folds': folds, 
        'test_auc': all_test_auc, 
        'test_acc': all_test_acc, 
        'test_f1': all_test_f1
    })
    
    result_df = pd.DataFrame({
        'metric': ['mean', 'var'],
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
    print("开始调用main函数...", flush=True)
    results, all_test_auc, all_test_acc, all_test_f1 = main(args)
    print("\n最终结果:")
    print(results)
    
    # 发送邮件通知
    if args.email_notification and args.recipient_email and args.sender_email and args.sender_password:
        results_summary = {
            'test_auc_mean': np.mean(all_test_auc) if len(all_test_auc) > 0 else 'N/A',
            'test_acc_mean': np.mean(all_test_acc) if len(all_test_acc) > 0 else 'N/A',
            'test_f1_mean': np.mean(all_test_f1) if len(all_test_f1) > 0 else 'N/A',
            'results_dir': args.results_dir,
            'task': args.task,
            'body_part': args.body_part if args.body_part else '全部'
        }
        
        # 尝试三次发送邮件，增加可靠性
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"尝试发送邮件通知 (第{attempt+1}次尝试)...")
                send_email_notification(
                    args.recipient_email,
                    args.smtp_server,
                    args.smtp_port,
                    args.sender_email,
                    args.sender_password,
                    args.exp_code,
                    results_summary
                )
                print("邮件发送成功!")
                break  # 成功发送后跳出循环
            except Exception as e:
                print(f"第{attempt+1}次尝试发送邮件失败: {e}")
                if attempt < max_retries - 1:
                    print(f"等待5秒后重试...")
                    time.sleep(5)
                else:
                    print("所有尝试均失败，无法发送邮件通知。")
    
    print("\n身体部位感知训练完成!")
    print("=======================================")
    print("主要改进:")
    print("1. 根据身体部位动态选择文本提示词")
    print("2. 仿照原版的双尺度并行匹配机制")
    print("3. 训练时精确匹配，测试时二分类")
    print("=======================================")