import torch.nn as nn
import torch
import timm
import os
import torch.nn.functional as F
from einops import rearrange

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)



class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1, bias=False)  # Conv+BN不需要bias
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class TimmEncoder(nn.Module):
    """
    使用timm库的预训练编码器
    支持更多预训练模型选择
    """
    
    def __init__(self, model_name='mobilenetv2_100', pretrained=True):
        super(TimmEncoder, self).__init__()
        
        self.model_name = model_name
        
        # 使用timm创建预训练模型
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            features_only=True,  # 只返回特征，不包含分类头
            out_indices=(0, 1, 2, 3, 4)  # 返回5个不同尺度的特征
        )

        # 获取特征通道数
        self.feature_channels = self.backbone.feature_info.channels()
    
    def get_feature_channels(self):
        """获取各层特征通道数"""
        return self.feature_channels
    
    def forward(self, x):
        """前向传播，返回多尺度特征"""

        features = self.backbone(x)
        
        return features

    

class Group_Spatial_att(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()

        self.channels = channels
        self.groups = groups

        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. 使用 rearrange 或 reshape 将通道分组
        # [B, C, H, W] -> [B, Groups, C_per_group, H, W]
        x_reshaped = x.reshape(b, self.groups, c // self.groups, h, w)
        
        # 2. 在组内计算通道维度的 max 和 avg -> [B, Groups, 1, H, W]
        avg_out = torch.mean(x_reshaped, dim=2, keepdim=True)
        max_out = torch.max(x_reshaped, dim=2, keepdim=True)[0]
        
        # 3. 拼接 -> [B, Groups, 2, H, W]
        att_input = torch.cat([max_out, avg_out], dim=2)
        
        # 4. 合并 B 和 Groups 维度以便进行共享卷积 -> [B*Groups, 2, H, W]
        att_input = att_input.reshape(b * self.groups, 2, h, w)
        
        # 5. 共享卷积 -> [B*Groups, 1, H, W]
        att = self.conv(att_input)
        att = self.sigmoid(att)
        
        # 6. 恢复形状 -> [B, Groups, H, W] 
        att = att.reshape(b, self.groups, h, w)
        
        return att

class MutiScale_Patch_fft(nn.Module):
    def __init__(self, channels, groups=4, patch_size=7):
        super().__init__()
        # 多尺度频域增强模块
        self.channels = channels
        self.patch_size = patch_size
        self.groups = groups
        
        # 三个分支的Patch FFT处理
        self.Patch_fft1 = Patch_fft(channels=channels, groups=groups, patch_size=patch_size)
        self.Patch_fft2 = Patch_fft(channels=channels, groups=groups, patch_size=patch_size)
        self.Patch_fft3 = Patch_fft(channels=channels, groups=groups, patch_size=patch_size)

        # 多尺度卷积分支
        # 分支1：3x3组卷积
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 分支2：5x5组卷积
        self.conv5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # 分支3：7x7组卷积
        self.conv7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, bias=False, groups=groups)
        self.bn3 = nn.BatchNorm2d(channels)
        
        # 统一激活函数
        self.act = nn.SiLU(inplace=True)

        # 自适应分支权重学习
        self.branch_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 3, kernel_size=1, bias=True),
        )

        # 特征融合（conv+bn组合）
        self.conv_fuse = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.bn_fuse = nn.BatchNorm2d(channels)
        
        # 残差连接
        self.use_residual = True
        
    def forward(self, x):
        identity = x
        
        # 三个分支：不同感受野的卷积 + BN + 激活 + FFT处理
        # 分支1：3x3组卷积
        x1 = self.act(self.bn1(self.conv3x3(x)))
        x1 = self.Patch_fft1(x1)
        
        # 分支2：5x5组卷积
        x2 = self.act(self.bn2(self.conv5x5(x)))
        x2 = self.Patch_fft2(x2)
        
        # 分支3：7x7组卷积
        x3 = self.act(self.bn3(self.conv7x7(x)))
        x3 = self.Patch_fft3(x3)

        # 拼接用于学习权重
        x_cat = torch.cat([x1, x2, x3], dim=1)  # [B, 3C, H, W]
        
        # 学习自适应分支权重
        logits = self.branch_weight(x_cat)  # [B, 3, 1, 1]
        weights = torch.softmax(logits, dim=1)  # [B, 3, 1, 1]
        
        # 加权融合三个分支
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        y = x1 * w1 + x2 * w2 + x3 * w3  # [B, C, H, W]
        
        # 特征融合 + BN + 激活
        y = self.bn_fuse(self.conv_fuse(y))
        y = self.act(y)
        
        # 残差连接
        if self.use_residual:
            y = y + identity

        return y



class Patch_fft(nn.Module):
    def __init__(self, channels, groups=4, patch_size=8):
        super().__init__()
        # 你的模块初始化代码
        self.channels = channels
        self.patch_size = patch_size
        self.groups = groups
        #self.spatial_att = Spatial_att2()

        self.group_spatial_att = Group_Spatial_att(channels=channels, groups=groups)

        self.freq_weight = nn.Parameter(
            torch.ones(channels, 1, 1, patch_size, patch_size // 2 + 1)
        )

        

    def forward(self, x):
        b, c, h, w = x.shape
        ps = self.patch_size



        if (h % ps != 0) or (w % ps != 0):
            raise ValueError(f"H({h}) 与 W({w}) 必须能被 patch_size({ps}) 整除。")

        # 空间按 patch 分块 -> (B, C, H/ps, W/ps, ps, ps)
        x_patch = rearrange(
            x, 'b c (hh ps1) (ww ps2) -> b c hh ww ps1 ps2',
            ps1=ps, ps2=ps
        )

        # x_patch shape: [B, C, hh, ww, ps, ps]
        x_patch_avg = x_patch.mean(dim=(-2, -1), keepdim=False)  # 空间池化
        patch_att = self.group_spatial_att(x_patch_avg)

        # (B, C_per_group, hh, ww, ps, ps)
        patch_att = patch_att.unsqueeze(-1).unsqueeze(-1)  # -> [B, groups, hh, ww, 1, 1]

        x_patch = x_patch.reshape(b, self.groups, c // self.groups, x_patch.shape[2], x_patch.shape[3], ps, ps)
        # 2D 实数 FFT 到频域（复数）
        x_patch_fft = torch.fft.rfft2(x_patch.float())  # complex64

        B, G, Cg, hh, ww, ps_dim, freq_dim = x_patch_fft.shape
        fw = self.freq_weight.view(self.groups, self.channels // self.groups, 1, 1, ps_dim, freq_dim)
        fw = fw.unsqueeze(0).to(x_patch_fft.device)  # [1, groups, Cg, 1,1, ps, freq]

        # 逐通道、逐频率分量可学习缩放（广播到 B、groups、Cg、hh、ww）
        x_patch_fft = x_patch_fft * fw  # 实数权重缩放复数谱

        # 逆 FFT 回到空间域（需指定原始大小）
        x_patch_filtered = torch.fft.irfft2(x_patch_fft, s=(ps, ps))

        x_patch_filtered = x_patch_filtered * patch_att.unsqueeze(2)

        # 还原为原图形状 -> (B, C, H, W)
        x_out = rearrange(
            x_patch_filtered, 'b num_groups c_groups hh ww ps1 ps2 -> b (num_groups c_groups) (hh ps1) (ww ps2)',
            ps1=ps, ps2=ps
        )


        return x_out

    



class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, mid_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels,(2,2),2)
        self.nConvs = _make_nConv(mid_channels, out_channels, nb_Conv, activation)

        
    def forward(self, x, skip_x=None):
        x = self.up(x)

        if skip_x is not None:
            x = torch.cat([x, skip_x], dim=1)  
            x = self.nConvs(x)
            out = x

        else:
            x = self.nConvs(x)
            out = x

        return out



class CLSF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv_h_mean = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, bias=True)
        self.conv_w_mean = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, bias=True)
        self.conv_h_max = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, bias=True)
        self.conv_w_max = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, bias=True)


        self.alpha = nn.Parameter(torch.ones(3))

        self.share_mlp_mean = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 16, 1, bias=True),
            nn.ReLU(),
            nn.Conv1d(in_channels // 16, in_channels, 1, bias=True),
        )
        self.share_mlp_max = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 16, 1, bias=True),
            nn.ReLU(),
            nn.Conv1d(in_channels // 16, in_channels, 1, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, feat1, feat2):
        # 直接在融合后的 feat 上做坐标注意
        B, C, H, W = x.shape

        # 稳定的三层软加权
        w = torch.softmax(self.alpha, dim=0)
        feat = w[0]*feat1 + w[1]*feat2 + w[2]*x

        h_cat = torch.cat([x, feat1, feat2], dim=2)
        w_cat = torch.cat([x, feat1, feat2], dim=3)

        # 沿宽度聚合 -> H 方向一维向量；沿高度聚合 -> W 方向一维向量
        h_mean = torch.mean(h_cat, dim=2)                 # (B,C,W)
        w_mean = torch.mean(w_cat, dim=3)                 # (B,C,H)
        h_max  = torch.max(h_cat,  dim=2).values          # (B,C,W)
        w_max  = torch.max(w_cat,  dim=3).values          # (B,C,H)

        
        # 用 1D conv 引入轴向上下文
        w_mean = self.conv_w_mean(w_mean)   # (B,C,H)
        h_mean = self.conv_h_mean(h_mean)   # (B,C,W)
        w_max  = self.conv_w_max(w_max)    # (B,C,H)
        h_max  = self.conv_h_max(h_max)    # (B,C,W)


        # 先按 [W, H] 顺序拼接，方便之后 split([W,H])
        hw_mean = torch.cat([h_mean, w_mean], dim=2)     # (B,C,W+H)
        hw_max  = torch.cat([h_max, w_max], dim=2)     # (B,C,W+H)

        hw_mean = self.share_mlp_mean(hw_mean)
        hw_max  = self.share_mlp_max(hw_max)

        # 正确顺序切分：[W, H]
        h_mean, w_mean= torch.split(hw_mean, [W, H], dim=2)
        h_max,  w_max  = torch.split(hw_max,  [W, H], dim=2)

        # 外积重建 (B,C,H,W)
        h_mean = h_mean.unsqueeze(-2)  # (B,C,1,W)
        w_mean = w_mean.unsqueeze(-1)  # (B,C,H,1)
        h_max  = h_max.unsqueeze(-2)
        w_max  = w_max.unsqueeze(-1)

        hw_mean_map = h_mean * w_mean
        hw_max_map  = h_max  * w_max

        out = self.sigmoid(hw_mean_map) * feat + self.sigmoid(hw_max_map) * feat

        return out

    
class CLCF(nn.Module):
    def __init__(self, x_channels):
        super().__init__()
        # MLP中的Linear后面没有BN，需要bias=True
        self.mlp = nn.Sequential(
            nn.Linear(x_channels*3, x_channels*3 // 16, bias=True),
            nn.ReLU(),
            nn.Linear(x_channels*3 // 16, x_channels*3, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, feat1, feat2):
        B, C, H, W = x.shape
        c_cat = torch.cat([x, feat1, feat2], dim=1)      # (B, 3C, H, W)
        avg = nn.functional.adaptive_avg_pool2d(c_cat, 1)
        mx  = nn.functional.adaptive_max_pool2d(c_cat, 1)

        att = self.mlp((avg + mx).view(B, -1)).view(B, 3, C, 1, 1)

        # 逐支路逐通道加权后再求和
        c_cat = c_cat.view(B, 3, C, H, W)
        out = torch.sum(att * c_cat, dim=1)

        return out
    
class ThreeFuseAttention(nn.Module):
    def __init__(self, feat1_channels, feat2_channels, x_channels):
        super().__init__()

        self.pwconv1 = nn.Conv2d(feat1_channels, x_channels, 1, bias=True)
        self.pwconv2 = nn.Conv2d(feat2_channels, x_channels, 1, bias=True)


        self.clsf = CLSF(x_channels)
        self.clcf  = CLCF(x_channels)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(x_channels * 2, x_channels * 2, 1, padding=0, bias=False),
            nn.BatchNorm2d(x_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(x_channels * 2, x_channels, 1, bias=True),
        )

    def forward(self, x, feat1, feat2):
        feat1 = self.pwconv1(feat1)
        feat1 = nn.functional.interpolate(feat1, size=x.shape[2:], mode='bilinear', align_corners=False)

        feat2 = self.pwconv2(feat2)
        feat2 = nn.functional.interpolate(feat2, size=x.shape[2:], mode='bilinear', align_corners=False)

        x1  = self.clsf(x, feat1, feat2)
        x2 = self.clcf(x, feat1, feat2)

        out = self.fuse_conv(torch.cat([x1, x2], dim=1))

        return out

class FESA_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, use_mobile_blocks=True, 
                 timm_model_name='mobilenetv2_100', pretrained=True):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        use_mobile_blocks : 是否使用MobileNet风格的块
        timm_model_name : timm模型名称，如'mobilenetv2_100', 'efficientnet_b0'等
        pretrained : 是否使用预训练权重
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.timm_model_name = timm_model_name
        self.use_pretrained = pretrained
        
        if n_classes != 1:
            self.n_classes = n_classes + 1
 

        self.encoder = TimmEncoder(timm_model_name, pretrained)
        feature_channels = self.encoder.get_feature_channels()
        self.feature_channels = feature_channels

        self.three_fuse1 = ThreeFuseAttention(feature_channels[-3], feature_channels[-1], feature_channels[-2])
        self.three_fuse2 = ThreeFuseAttention(feature_channels[-4], feature_channels[-2], feature_channels[-3])
        self.three_fuse3 = ThreeFuseAttention(feature_channels[-5], feature_channels[-3], feature_channels[-4])


        self.multiscale_patch_fft1 = MutiScale_Patch_fft(channels=feature_channels[-2], groups=4, patch_size=7)
        self.multiscale_patch_fft2 = MutiScale_Patch_fft(channels=feature_channels[-3], groups=4, patch_size=7)
        self.multiscale_patch_fft3 = MutiScale_Patch_fft(channels=feature_channels[-4], groups=4, patch_size=7)
        self.multiscale_patch_fft4 = MutiScale_Patch_fft(channels=feature_channels[-5], groups=4, patch_size=7)
    
        # 解码器部分 - 使用MobileNet风格块
        decoder_channels = feature_channels[-1]

        # 修改解码器结构以匹配5个特征层
        self.up5 = UpBlock(decoder_channels, decoder_channels + feature_channels[-2], feature_channels[-3], nb_Conv=2)
        self.up4 = UpBlock(feature_channels[-3], feature_channels[-3]*2, feature_channels[-4], nb_Conv=2)
        self.up3 = UpBlock(feature_channels[-4], feature_channels[-4]*2, feature_channels[-5], nb_Conv=2)
        self.up2 = UpBlock(feature_channels[-5], feature_channels[-5]*2, feature_channels[-5], nb_Conv=2)
        self.up1 = UpBlock(feature_channels[-5], feature_channels[-5], feature_channels[-5], nb_Conv=2)

        self.outc = nn.Conv2d(feature_channels[-5], self.n_classes, kernel_size=(1,1), bias=True)
            
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):


        # 使用timm预训练编码器
        encoder_features = self.encoder(x)

        x = encoder_features[-1] 
        
        # 解码器部分
        y = self.three_fuse1(encoder_features[-2], encoder_features[-3], encoder_features[-1]) 
        y = self.multiscale_patch_fft1(y)
        x = self.up5(x, y)
        y = self.three_fuse2(encoder_features[-3], encoder_features[-4], encoder_features[-2])
        y = self.multiscale_patch_fft2(y)
        x = self.up4(x, y)
        y = self.three_fuse3(encoder_features[-4], encoder_features[-5], encoder_features[-3])
        y = self.multiscale_patch_fft3(y)
        x = self.up3(x, y)
        y = self.multiscale_patch_fft4(encoder_features[-5])
        x = self.up2(x, y)
        x = self.up1(x, None)

        # 最终卷积和激活
        x = self.outc(x)
        

        if self.last_activation is not None:
            x = self.last_activation(x)
            
        return x

