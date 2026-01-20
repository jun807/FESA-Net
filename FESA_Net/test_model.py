"""
医学图像分割模型测试脚本


主要功能：
1. 加载训练好的模型进行测试
2. 计算Dice系数和IoU指标
3. 生成可视化结果和热力图
4. 支持多种数据集和模型的测试
"""

import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
import pickle
# from nets.Unet22_2_3_2_grp import Unet22_2_3_2_grp

import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os

# 导入各种分割模型
# from nets.ACC_UNet import ACC_UNet
# from nets.UCTransNet import UCTransNet
# from nets.UNet_base import UNet_base
# from nets.SMESwinUnet import SMESwinUnet
# from nets.MResUNet1 import MultiResUnet
# from nets.SwinUnet import SwinUnet
# from nets.SwinPA import SwinPA

from nets.FESA_Net import FESA_Net

import json
from utils import *
import cv2


def show_image_with_dice(predict_save, labs, save_path):
    """
    计算预测结果与真实标签之间的Dice系数和IoU
    
    Args:
        predict_save: 模型预测结果（二值化后的分割图）
        labs: 真实标签（ground truth）
        save_path: 保存路径（当前未使用）
        
    Returns:
        dice_pred: Dice系数
        iou_pred: IoU（交并比）
    """
    # 转换为浮点数类型
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    
    # 计算Dice系数：2*交集/(集合A+集合B)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    
    # 计算IoU（交并比）
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))

    return dice_pred, iou_pred
    

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    """
    可视化预测结果并保存热力图
    
    该函数执行模型推理，计算评估指标，并保存可视化结果
    
    Args:
        model: 训练好的模型
        input_img: 输入图像
        img_RGB: RGB图像（当前未使用）
        labs: 真实标签
        vis_save_path: 可视化结果保存路径
        dice_pred: 累积的Dice分数
        dice_ens: 累积的集成Dice分数
        
    Returns:
        dice_pred_tmp: 当前图像的Dice分数
        iou_tmp: 当前图像的IoU分数
    """
    model.eval()  # 设置为评估模式

    # 模型推理
    output = model(input_img.cuda())
    
    # 二值化预测结果（阈值0.5）
    pred_class = torch.where(output>0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    
    # 计算Dice和IoU指标
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    input_img.to('cpu')  # 将输入图像移回CPU

    # 准备数据用于保存和可视化
    input_img = input_img[0].transpose(0,-1).cpu().detach().numpy()  # 输入图像
    labs = labs[0]  # 真实标签
    output = output[0,0,:,:].cpu().detach().numpy()  # 模型输出

    # 保存预测结果到pickle文件（用于后续分析）
    if(True):
        pickle.dump({
            'input': input_img,  # 输入图像
            'output': (output>=0.5)*1.0,  # 二值化后的预测结果
            'ground_truth': labs,  # 真实标签
            'dice': dice_pred_tmp,  # Dice分数
            'iou': iou_tmp  # IoU分数
        },
        open(vis_save_path+'.p', 'wb'))

    # 生成可视化图像
    if(True):
        # 创建包含输入、真实标签和预测结果的对比图
        plt.figure(figsize=(10,3.3))
        plt.subplot(1,3,1)
        # 优先使用原始RGB图像
        if img_RGB is not None:
            if isinstance(img_RGB, torch.Tensor):
                img_to_show = img_RGB.cpu().numpy().transpose(1,2,0)
            elif hasattr(img_RGB, 'numpy'):
                img_to_show = np.array(img_RGB)
            else:
                img_to_show = img_RGB
            plt.imshow(img_to_show.astype(np.uint8))
        else:
            plt.imshow(input_img)
        plt.subplot(1,3,2)
        plt.imshow(labs, cmap='gray')  # 真实标签
        plt.subplot(1,3,3)
        plt.imshow((output>=0.5)*1.0, cmap='gray')  # 预测结果
        plt.suptitle(f'Dice score : {np.round(dice_pred_tmp,3)}\nIoU : {np.round(iou_tmp,3)}')
        plt.tight_layout()
        plt.savefig(vis_save_path)
        plt.close()

        # 分别保存输入图、标签图和预测图
        name_no_ext = os.path.splitext(vis_save_path)[0]
        # 保存输入图
        if img_RGB is not None:
            if isinstance(img_RGB, torch.Tensor):
                img_to_save = img_RGB.cpu().numpy().transpose(1,2,0)
            elif hasattr(img_RGB, 'numpy'):
                img_to_save = np.array(img_RGB)
            else:
                img_to_save = img_RGB
            plt.imsave(name_no_ext + '_input.png', img_to_save.astype(np.uint8))
        else:
            if input_img.shape[-1] == 1:
                plt.imsave(name_no_ext + '_input.png', input_img.squeeze(), cmap='gray')
            else:
                plt.imsave(name_no_ext + '_input.png', input_img)
        # 保存标签图
        plt.imsave(name_no_ext + '_gt.png', labs, cmap='gray')
        # 保存预测图
        plt.imsave(name_no_ext + '_pred.png', (output>=0.5)*1.0, cmap='gray')

    return dice_pred_tmp, iou_tmp


if __name__ == '__main__':
    """
    主程序入口
    
    根据配置加载模型，对测试集进行推理，计算评估指标并保存结果
    """
    
    # 设置GPU设备（当前被注释掉）
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # =============================================================
    #               根据任务配置测试参数
    # =============================================================
    test_session = config.test_session
    
    # GlaS数据集配置（结肠腺癌分割）
    if config.task_name == "GlaS_exp1":
        test_num = 80  # 测试图像数量
        model_type = config.model_name
        model_path = "./GlaS_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "GlaS_exp2":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "GlaS_exp3":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    # ISIC18数据集配置（皮肤病变分割）
    elif config.task_name == "ISIC18_exp1":
        test_num = 1000
        model_type = config.model_name
        model_path = "./ISIC18_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "ISIC18_exp2":
        test_num = 1000
        model_type = config.model_name
        model_path = "./ISIC18_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "ISIC18_exp3":
        test_num = 1000
        model_type = config.model_name
        model_path = "./ISIC18_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "ISIC17_exp1":
        test_num = 600
        model_type = config.model_name
        model_path = "./ISIC17_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    
    elif config.task_name == "ISIC17_exp2":
        test_num = 600
        model_type = config.model_name
        model_path = "./ISIC17_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "ISIC17_exp3":
        test_num = 600
        model_type = config.model_name
        model_path = "./ISIC17_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "DSB2018_exp1":
        test_num = 68
        model_type = config.model_name
        model_path = "./DSB2018_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "DSB2018_exp2":
        test_num = 68
        model_type = config.model_name
        model_path = "./DSB2018_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "DSB2018_exp3":
        test_num = 68
        model_type = config.model_name
        model_path = "./DSB2018_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
        
    # ClinicDB数据集配置（息肉分割）
    elif config.task_name == "Clinic_exp1":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "Clinic_exp2":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "Clinic_exp3":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    
    # BUSI数据集配置（乳腺超声图像分割）
    elif config.task_name == "BUSI_exp1":
        test_num = 120
        model_type = config.model_name
        model_path = "./BUSI_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "BUSI_exp2":
        test_num = 120
        model_type = config.model_name
        model_path = "./BUSI_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "BUSI_exp3":
        test_num = 120
        model_type = config.model_name
        model_path = "./BUSI_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    
    # COVID数据集配置（COVID-19肺部感染分割）
    elif config.task_name == "Covid_exp1":
        test_num = 169
        model_type = config.model_name
        model_path = "./Covid_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "Covid_exp2":
        test_num = 169
        model_type = config.model_name
        model_path = "./Covid_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "Covid_exp3":
        test_num = 169
        model_type = config.model_name
        model_path = "./Covid_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
        
    elif config.task_name == "Kvasir_exp1":
        test_num = 200
        model_type = config.model_name
        model_path = "./Kvasir_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "Kvasir_exp2":
        test_num = 200
        model_type = config.model_name
        model_path = "./Kvasir_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "Kvasir_exp3":
        test_num = 200
        model_type = config.model_name
        model_path = "./Kvasir_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "ph2_exp1":
        test_num = 30
        model_type = config.model_name
        model_path = "./ph2_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar" 

    elif config.task_name == "ph2_exp2":
        test_num = 30
        model_type = config.model_name
        model_path = "./ph2_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar" 

    elif config.task_name == "ph2_exp3":
        test_num = 30
        model_type = config.model_name
        model_path = "./ph2_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar" 

    
    # =============================================================
    #               设置保存路径和创建目录
    # =============================================================
    save_path = config.task_name + '/' + config.model_name + '/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    vis_path = save_path + 'visualize_test/'
    
    # 创建可视化结果保存目录
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    # =============================================================
    #               加载训练好的模型
    # =============================================================
    checkpoint = torch.load(model_path, map_location='cuda')


    # 打开结果文件，记录测试开始时间
    fp = open(save_path+'test.result', 'a')
    fp.write(str(datetime.now())+'\n')

    # =============================================================
    #               根据模型类型初始化模型
    # =============================================================
    if model_type == 'ACC_UNet':
        # ACC-UNet模型
        config_vit = config.get_CTranS_config()   
        model = ACC_UNet(n_channels=config.n_channels, n_classes=config.n_labels, n_filts=config.n_filts)

    elif model_type == 'UCTransNet':
        # UCTransNet模型（基于Transformer）
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNet_base':
        # 基础UNet模型
        config_vit = config.get_CTranS_config()   
        model = UNet_base(n_channels=config.n_channels, n_classes=config.n_labels)
        
    elif model_type == 'SwinUnet':            
        # SwinUnet模型
        model = SwinUnet()
        model.load_from()  # 加载预训练权重

    elif model_type == 'SMESwinUnet':            
        # SMESwinUnet模型
        model = SMESwinUnet(n_channels=config.n_channels, n_classes=config.n_labels)
        model.load_from()  # 加载预训练权重

    elif model_type.split('_')[0] == 'MultiResUnet1':          
        # MultiResUnet模型，支持自定义参数
        model = MultiResUnet(n_channels=config.n_channels, n_classes=config.n_labels, 
                           nfilt=int(model_type.split('_')[1]), alpha=float(model_type.split('_')[2]))
    
    elif model_type == 'SwinPA':
        # SwinPA模型
        model = SwinPA(n_channels=config.n_channels, n_classes=config.n_labels, encoder='inceptionnext_base')
        model.load_from()  # 加载预训练权重

        
    elif model_type == 'FESA_Net':  

        config_vit = config.get_CTranS_config()        
        model = FESA_Net(n_channels=config.n_channels,n_classes=config.n_labels,use_mobile_blocks=config.use_mobile_blocks,timm_model_name='mobilenetv2_100',pretrained=config.pretrained)
               
    
    else: 
        raise TypeError('Please enter a valid name for the model type')

    # =============================================================
    #               模型部署和权重加载
    # =============================================================
    model = model.cuda()  # 将模型移动到GPU
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    
    # 加载训练好的权重
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Model loaded !')
    
    # =============================================================
    #               准备测试数据
    # =============================================================
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 测试时不打乱数据

    # =============================================================
    #               开始测试
    # =============================================================
    dice_pred = 0.0  # 累积Dice分数
    iou_pred = 0.0   # 累积IoU分数
    dice_ens = 0.0   # 累积集成Dice分数

    # 使用进度条显示测试进度
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            # 获取测试数据和标签
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            
            # 数据预处理
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            
            # 以下代码被注释掉，原本用于保存真实标签图像
            ###########fig, ax = plt.subplots()
            ###########plt.imshow(img_lab, cmap='gray')
            ###########plt.axis("off")
            height, width = config.img_size, config.img_size
            ###########fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            ###########plt.gca().xaxis.set_major_locator(plt.NullLocator())
            ###########plt.gca().yaxis.set_major_locator(plt.NullLocator())
            ###########plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ###########plt.margins(0, 0)
            ###########plt.savefig(vis_path+str(i)+"_lab.jpg", dpi=300)
            ###########plt.close()
            
            # 转换为PyTorch张量
            input_img = torch.from_numpy(arr)
            
            # 进行模型推理和可视化
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(i)+'.png',
                                                          dice_pred=dice_pred, dice_ens=dice_ens)
            
            # 累积评估指标
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            pbar.update()  # 更新进度条
    
    # =============================================================
    #               输出最终结果
    # =============================================================
    # 计算平均指标
    print("dice_pred", dice_pred/test_num)
    print("iou_pred", iou_pred/test_num)
    
    # 将结果写入文件
    fp.write(f"dice_pred : {dice_pred/test_num}\n")
    fp.write(f"iou_pred : {iou_pred/test_num}\n")
    fp.close()



