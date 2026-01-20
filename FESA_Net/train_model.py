"""
医学图像分割模型训练脚本

主要功能：
1. 支持多种分割模型的训练
2. 提供完整的训练、验证和模型保存流程
3. 支持早停机制和学习率调度
4. 集成TensorBoard日志记录
"""

import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D

# 导入各种分割模型
# from nets.ACC_UNet import ACC_UNet
# from nets.MResUNet1 import MultiResUnet
# from nets.SwinUnet import SwinUnet
# from nets.UNet_base import UNet_base
# from nets.SMESwinUnet import SMESwinUnet
# from nets.UCTransNet import UCTransNet
# from nets.SwinPA import SwinPA

from nets.FESA_Net import FESA_Net

from torch.utils.data import DataLoader
import logging
import json
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE

def logger_config(log_path):
    """
    配置日志记录器
    
    设置日志格式、输出位置和控制台输出，用于记录训练过程中的信息
    
    Args:
        log_path (str): 日志文件保存路径
        
    Returns:
        logger: 配置好的日志记录器
    """
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    
    # 文件处理器：将日志写入文件
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    # 控制台处理器：在控制台显示日志
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    # 添加处理器到日志记录器
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def save_checkpoint(state, save_path):
    """
    保存模型检查点
    
    根据是否为最佳模型选择不同的保存策略：
    - 最佳模型：保存为 'best_model-{model_name}.pth.tar'
    - 普通检查点：保存为 'model-{model_name}-{epoch}.pth.tar'
    
    Args:
        state (dict): 包含模型状态、优化器状态等信息的字典
        save_path (str): 模型保存路径
    """
    logger.info('\t Saving to {}'.format(save_path))
    
    # 创建保存目录（如果不存在）
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # 当前训练轮数
    best_model = state['best_model']  # 是否为最佳模型
    model = state['model']  # 模型类型

    if best_model:
        # 保存最佳模型
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        # 保存普通检查点
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)

def worker_init_fn(worker_id):
    """
    数据加载器工作进程初始化函数
    
    为每个工作进程设置不同的随机种子，确保多进程数据加载的随机性
    
    Args:
        worker_id (int): 工作进程ID
    """
    random.seed(config.seed + worker_id)

##################################################################################
#=================================================================================
#          主循环：加载模型、训练、验证
#=================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    """
    主训练循环
    
    包含完整的模型训练流程：数据加载、模型初始化、训练、验证、模型保存
    
    Args:
        batch_size (int): 批次大小，默认使用配置文件中的值
        model_type (str): 模型类型名称
        tensorboard (bool): 是否启用TensorBoard日志记录
        
    Returns:
        model: 训练完成的模型
    """
    # =============================================================
    #               数据加载器设置
    # =============================================================
    
    # 定义数据增强和预处理
    train_tf= transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    
    # 创建训练和验证数据集
    train_dataset = ImageToImage2D(config.train_dataset, train_tf,image_size=config.img_size)
    val_dataset = ImageToImage2D(config.val_dataset, val_tf,image_size=config.img_size)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,  # 训练时打乱数据
                              worker_init_fn=worker_init_fn,
                              num_workers=8,  # 使用8个工作进程
                              pin_memory=True)  # 使用固定内存加速GPU传输
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,  # 验证时也打乱数据
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)

    # 设置学习率
    lr = config.learning_rate
    
    # 记录模型信息
    logger.info(model_type)
    logger.info('n_filts : ' + str(config.n_filts))

    # =============================================================
    #               模型初始化
    # =============================================================

    if model_type == 'UCTransNet':
        # UCTransNet模型（基于Transformer）
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = UCTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'UNet_base':
        # 基础UNet模型
        config_vit = config.get_CTranS_config()        
        model = UNet_base(n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'SMESwinUnet':
        # SMESwinUnet模型
        config_vit = config.get_CTranS_config()        
        model = SMESwinUnet(n_channels=config.n_channels,n_classes=config.n_labels)
        model.load_from()  # 加载预训练权重
        lr = 5e-4  # 使用较小的学习率

    elif model_type == 'SwinUnet':            
        # SwinUnet模型
        model = SwinUnet()
        model.load_from()  # 加载预训练权重
        lr = 5e-4  # 使用较小的学习率
    
    elif model_type == 'SwinPA':
        # SwinPA模型
        model = SwinPA(n_channels=config.n_channels, n_classes=config.n_labels, encoder='inceptionnext_base')
        model.load_from()  # 加载预训练权重
        lr = 5e-4  # 使用较小的学习率

    elif model_type.split('_')[0] == 'MultiResUnet1':          
        # MultiResUnet模型，支持自定义参数
        model = MultiResUnet(n_channels=config.n_channels,n_classes=config.n_labels,nfilt=int(model_type.split('_')[1]), alpha=float(model_type.split('_')[2]))        
    
    
    elif model_type == 'FESA_Net':  

        config_vit = config.get_CTranS_config()        
        model = FESA_Net(n_channels=config.n_channels,n_classes=config.n_labels,use_mobile_blocks=config.use_mobile_blocks,timm_model_name='mobilenetv2_100',pretrained=config.pretrained)
               

    else: 
        raise TypeError('Please enter a valid name for the model type')

    # =============================================================
    #               优化器设置
    # =============================================================
    
    if model_type == 'SwinUnet':            
        # SwinUnet使用SGD优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    elif model_type == 'SMESwinUnet':
        # SMESwinUnet使用SGD优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    else:
        # 其他模型使用Adam优化器
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    # 将模型移动到GPU
    model = model.cuda()

    # 记录训练环境信息
    logger.info('Training on ' +str(os.uname()[1]))
    logger.info('Training using GPU : '+torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # 多GPU支持（当前被注释掉）
    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])
    
    # =============================================================
    #               损失函数和调度器设置
    # =============================================================
    
    # 使用加权Dice+BCE损失函数
    criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5, n_labels=config.n_labels)
    
    # 学习率调度器设置
    if config.cosineLR is True:
        # 使用余弦退火学习率调度
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None
        
    # =============================================================
    #               TensorBoard设置
    # =============================================================
    
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # =============================================================
    #               训练循环
    # =============================================================
    
    max_dice = 0.0  # 记录最佳Dice分数
    best_epoch = 1  # 记录最佳模型对应的轮数
    
    print("config.epochs",config.epochs)
    for epoch in range(config.epochs):  # 循环训练指定轮数
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        
        # 训练一个轮次
        model.train(True)  # 设置为训练模式
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)
        
        # 在验证集上评估
        logger.info('Validation')
        with torch.no_grad():  # 验证时不计算梯度
            model.eval()  # 设置为评估模式
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                            optimizer, writer, epoch, lr_scheduler,model_type,logger)

        # =============================================================
        #               保存最佳模型
        # =============================================================
        if val_dice > max_dice:
            # 如果当前验证Dice分数超过历史最佳，保存模型
            logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
            max_dice = val_dice
            best_epoch = epoch + 1
            save_checkpoint({'epoch': epoch,
                             'best_model': True,
                             'model': model_type,
                             'state_dict': model.state_dict(),
                             'val_loss': val_loss,
                             'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            # 如果Dice分数没有提升，记录信息
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        
        # 早停机制
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count,config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            # 如果连续多轮没有提升，提前停止训练
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    """
    主程序入口
    
    设置随机种子、创建必要目录、配置日志、开始训练
    """
    
    # =============================================================
    #               确定性设置
    # =============================================================
    deterministic = True
    if not deterministic:
        # 非确定性模式：允许CUDA优化，训练速度更快但结果可能略有不同
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        # 确定性模式：确保结果可重现，但训练速度较慢
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    # =============================================================
    #               随机种子设置
    # =============================================================
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # =============================================================
    #               目录创建
    # =============================================================
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    # =============================================================
    #               日志配置
    # =============================================================
    # 如果日志文件已存在，退出程序（防止覆盖）
    if os.path.isfile(config.logger_path):
        import sys
        sys.exit()
    logger = logger_config(log_path=config.logger_path)
    
    # =============================================================
    #               开始训练
    # =============================================================
    model = main_loop(model_type=config.model_name, tensorboard=True)
    
    # =============================================================
    #               训练完成记录
    # =============================================================
    fp = open('log.log','a')
    fp.write(f'{config.model_name} on {config.task_name} completed\n')
    fp.close()
    