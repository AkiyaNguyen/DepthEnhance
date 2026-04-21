import torch
import torch.nn as nn

import torchvision.models as models
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

from torchinfo import summary

import numpy as np
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x



class encoder(nn.Module):
    def __init__(self, num_classes):
        super(encoder, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):
        # x 224
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        return e1, e2, e3, e4, e5 ## 64, 64, 128, 256, 512



class encoder18(nn.Module):
    def __init__(self, num_classes):
        super(encoder18, self).__init__()


        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):

        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        return e1, e2, e3, e4, e5

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()


        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, feature):
        e1, e2, e3, e4, e5 = feature
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out1 = self.outconv(d1)  # 224

        return torch.sigmoid(out1)


class ResNet34U_f(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(ResNet34U_f, self).__init__()

        self.encoder1 = encoder(num_classes)

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )
        # Decoder stage channel widths for indices 1..5; proj_head built in set_feature_layers (not here)
        # so optimizers can be constructed after eager convs exist (avoids LazyConv2d / stale param refs).
        self.out_channel_list = [64, 64, 128, 256, 512]
        self.feature_layers = None
        self.proj_head = None

    def set_feature_layers(self, feature_layers: int, add_proj_head: bool = False,
            middle_channels: int = 128, out_channels: int = 256):
        if not 1 <= feature_layers <= 5:
            raise ValueError("feature_layers must be in [1,5]")
        self.feature_layers = feature_layers
        dev = next(self.parameters()).device
        if add_proj_head:
            c = self.out_channel_list[feature_layers - 1]
            self.proj_head = nn.Sequential(
                nn.Conv2d(c, middle_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(middle_channels, out_channels, kernel_size=1),
            ).to(dev)
        else:
            self.proj_head = None

    def forward(self, x, fp=False, type='decoder'):
        if fp and self.feature_layers is None:
            raise ValueError("feature_layers is not set, please set it before forward")

        e1, e2, e3, e4, e5 = self.encoder1(x)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out1 = self.outconv(d1)

        decoder_fea_layers = [None, d1, d2, d3, d4, d5]
        rgb_encoder_fea_layers = [None, e1, e2, e3, e4, e5]
        if fp:
            if type == 'decoder':
                fea = decoder_fea_layers[self.feature_layers]
            elif type == 'encoder':
                fea = rgb_encoder_fea_layers[self.feature_layers]
            else:
                raise ValueError(f"Invalid type: {type}, allowed types are 'decoder', 'encoder'")

            if self.proj_head is not None:
                fea = self.proj_head(fea)

            return torch.sigmoid(out1), fea
            
        else:
            return torch.sigmoid(out1)


class ResNet18U_f(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(ResNet18U_f, self).__init__()


        self.encoder1 = encoder18(num_classes)
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )


    def forward(self, x,fp=False):
        e1, e2, e3, e4, e5 = self.encoder1(x)

        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out1 = self.outconv(d1)
        if fp:
            return F.sigmoid(out1), e5
        else:
            return F.sigmoid(out1)


class CNNFusionBlock(nn.Module):
    def __init__(self, c1, c2, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1 + c2, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=1))

class SEBlock(nn.Module):
    def __init__(self,input_channels, mid_channels=32):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, mid_channels, 1), nn.ReLU(),
            nn.Conv2d(mid_channels, input_channels, 1), nn.Sigmoid()   
        )
    def forward(self, x):
        return x * self.se(x)

class SEFusionBlock(nn.Module):
    def __init__(self, c1, c2, out_c):
        super().__init__()
        total_c = c1 + c2
        self.se = SEBlock(total_c)
        self.conv = nn.Conv2d(total_c, out_c, 3, padding=1)
    
    def forward(self, x1, x2):
        cat = torch.cat([x1, x2], dim=1)  # [B,1024,H,W]
        
        enhanced = self.se(cat)
        
        return self.conv(enhanced)

class ACFusionBlock(nn.Module):
    """
    implementation of attention complementary module 
    from "https://arxiv.org/abs/1905.10089"

    """
    def __init__(self, channel):
        super().__init__()
        self.se1 = SEBlock(input_channels=channel)
        self.se2 = SEBlock(input_channels=channel)
    def forward(self, x1, x2, preceding_feature=None):
        """
        ensure that the shape of x1 and x2 are the same and previous feature (if exists) are the same
        """
        se1 = self.se1(x1)
        se2 = self.se2(x2)
        if preceding_feature is not None:
            return preceding_feature + se1 + se2
        else:
            return se1 + se2


# class Depth_W_ACM_ResNet34U_f_EMAEncoderOnly(nn.Module):
#     def __init__(self, num_classes, dropout=0.1):
#         super().__init__()
#         self.rgb_encoder = encoder(num_classes=None)

#         self.depth_encoder = encoder(num_classes=None)
        
#         # Fusion blocks
#         self.fusion_block5 = ACFusionBlock(512)
#         self.fusion_block4 = ACFusionBlock(256)
#         self.fusion_block3 = ACFusionBlock(128)
#         self.fusion_block2 = ACFusionBlock(64)
#         self.fusion_block1 = ACFusionBlock(64)

#         res_merge = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

#         self.merge_layer1 = res_merge.layer1
#         self.merge_layer2 = res_merge.layer2
#         self.merge_layer3 = res_merge.layer3
#         self.merge_layer4 = res_merge.layer4

#         self.decoder5 = DecoderBlock(512, 512)
#         self.decoder4 = DecoderBlock(512 + 256, 256)
#         self.decoder3 = DecoderBlock(256 + 128, 128)
#         self.decoder2 = DecoderBlock(128 + 64, 64)
#         self.decoder1 = DecoderBlock(64 + 64, 64)
        
#         self.outconv = nn.Sequential(
#             ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(32, num_classes, 1),
#         )
    
#     def forward(self, x, depth, fp=False):

#         # RGB-D fusion
#         e1, e2, e3, e4, e5 = self.rgb_encoder(x)
#         e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
        
#         # f5 = self.fusion_block5(e5, e5_d)
#         # f5 = ...
#         # f4 = self.fusion_block4(e4, e4_d, preceding_feature=f5)
#         # f4 = ...
#         # f3 = self.fusion_block3(e3, e3_d, preceding_feature=f4)
#         # f3 = ...
#         # f2 = self.fusion_block2(e2, e2_d, preceding_feature=f3)
#         # f2 = ...
#         # f1 = self.fusion_block1(e1, e1_d, preceding_feature=f2)

#         f1 = self.fusion_block1(e1, e1_d)
#         m = nn.MaxPool2d(3, stride=2, padding=1)
#         merge_f1 = self.merge_layer1(m(f1))
#         f2 = self.fusion_block2(e2, e2_d, preceding_feature=merge_f1)
#         merge_f2 = self.merge_layer2(f2)
#         f3 = self.fusion_block3(e3, e3_d, preceding_feature=merge_f2)
#         merge_f3 = self.merge_layer3(f3)
#         f4 = self.fusion_block4(e4, e4_d, preceding_feature=merge_f3)
#         merge_f4 = self.merge_layer4(f4)
#         f5 = self.fusion_block5(e5, e5_d, preceding_feature=merge_f4)

#         # Decoder
#         d5 = self.decoder5(f5)
#         d4 = self.decoder4(torch.cat([d5, f4], dim=1))
#         d3 = self.decoder3(torch.cat([d4, f3], dim=1))
#         d2 = self.decoder2(torch.cat([d3, f2], dim=1))
#         d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
#         out1 = self.outconv(d1)
#         final_output = F.sigmoid(out1)
#         if fp:
#             return final_output, f5
#         else:
#             return final_output



# class Depth_W_CNNFusion_ResNet34U_f(nn.Module):
#     def __init__(self, num_classes, dropout=0.1):
#         super().__init__()
#         self.rgb_branch = ResNet34U_f(num_classes, dropout)        
#         self.depth_encoder = encoder(num_classes=None)
        
#         # Fusion blocks
#         self.fusion_block5 = CNNFusionBlock(512, 512, 512)
#         self.fusion_block4 = CNNFusionBlock(256, 256, 256)
#         self.fusion_block3 = CNNFusionBlock(128, 128, 128)
#         self.fusion_block2 = CNNFusionBlock(64, 64, 64)
#         self.fusion_block1 = CNNFusionBlock(64, 64, 64)
        
#         self.decoder5 = DecoderBlock(512, 512)
#         self.decoder4 = DecoderBlock(512 + 256, 256)
#         self.decoder3 = DecoderBlock(256 + 128, 128)
#         self.decoder2 = DecoderBlock(128 + 64, 64)
#         self.decoder1 = DecoderBlock(64 + 64, 64)
        
#         self.outconv = nn.Sequential(
#             ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(32, num_classes, 1),
#         )
    
#     def forward(self, x, depth=None, fp=False):
#         e1, e2, e3, e4, e5 = self.rgb_branch.encoder1(x)
#         output = {'rgb': self.rgb_branch(x, fp=fp), 'rgb_depth': None}
        
#         if depth is None:
#             return output
        
#         # RGB-D fusion
#         e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
#         f5 = self.fusion_block5(e5, e5_d)
#         f4 = self.fusion_block4(e4, e4_d)
#         f3 = self.fusion_block3(e3, e3_d)
#         f2 = self.fusion_block2(e2, e2_d)
#         f1 = self.fusion_block1(e1, e1_d)
        
#         # Decoder
#         d5 = self.decoder5(f5)
#         d4 = self.decoder4(torch.cat([d5, f4], dim=1))
#         d3 = self.decoder3(torch.cat([d4, f3], dim=1))
#         d2 = self.decoder2(torch.cat([d3, f2], dim=1))
#         d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
#         out1 = self.outconv(d1)
#         output['rgb_depth'] = (F.sigmoid(out1), f5) if fp else F.sigmoid(out1)
        # return output


# class Depth_W_SEFusion_ResNet34U_f(nn.Module):
#     def __init__(self, num_classes, dropout=0.1):
#         super().__init__()
#         self.rgb_branch = ResNet34U_f(num_classes, dropout)        
#         self.depth_encoder = encoder(num_classes=None)
        
#         # Fusion blocks
#         self.fusion_block5 = SEFusionBlock(512, 512, 512)
#         self.fusion_block4 = SEFusionBlock(256, 256, 256)
#         self.fusion_block3 = SEFusionBlock(128, 128, 128)
#         self.fusion_block2 = SEFusionBlock(64, 64, 64)
#         self.fusion_block1 = SEFusionBlock(64, 64, 64)
        
#         self.decoder5 = DecoderBlock(512, 512)
#         self.decoder4 = DecoderBlock(512 + 256, 256)
#         self.decoder3 = DecoderBlock(256 + 128, 128)
#         self.decoder2 = DecoderBlock(128 + 64, 64)
#         self.decoder1 = DecoderBlock(64 + 64, 64)
        
#         self.outconv = nn.Sequential(
#             ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(32, num_classes, 1),
#         )
    
#     def forward(self, x, depth=None, fp=False):
#         output = {'rgb': self.rgb_branch(x, fp=fp), 'rgb_depth': None}
        
#         if depth is None:
#             return output
        
#         # RGB-D fusion
#         e1, e2, e3, e4, e5 = self.rgb_branch.encoder1(x)
#         e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
#         f5 = self.fusion_block5(e5, e5_d)
#         f4 = self.fusion_block4(e4, e4_d)
#         f3 = self.fusion_block3(e3, e3_d)
#         f2 = self.fusion_block2(e2, e2_d)
#         f1 = self.fusion_block1(e1, e1_d)
        
#         # Decoder
#         d5 = self.decoder5(f5)
#         d4 = self.decoder4(torch.cat([d5, f4], dim=1))
#         d3 = self.decoder3(torch.cat([d4, f3], dim=1))
#         d2 = self.decoder2(torch.cat([d3, f2], dim=1))
#         d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
#         out1 = self.outconv(d1)
#         output['rgb_depth'] = (F.sigmoid(out1), f5) if fp else F.sigmoid(out1)
#         return output


# class DepthFusion_ResNet34U_f_EMAEncoderOnly(nn.Module):
#     def __init__(self, num_classes, dropout=0.1):
#         super().__init__()
#         self.rgb_encoder = encoder(num_classes=None)

#         self.depth_encoder = encoder(num_classes=None)
        
#         # Fusion blocks
#         self.fusion_block5 = SEFusionBlock(512, 512, 512)
#         self.fusion_block4 = SEFusionBlock(256, 256, 256)
#         self.fusion_block3 = SEFusionBlock(128, 128, 128)
#         self.fusion_block2 = SEFusionBlock(64, 64, 64)
#         self.fusion_block1 = SEFusionBlock(64, 64, 64)
        
#         self.decoder5 = DecoderBlock(512, 512)
#         self.decoder4 = DecoderBlock(512 + 256, 256)
#         self.decoder3 = DecoderBlock(256 + 128, 128)
#         self.decoder2 = DecoderBlock(128 + 64, 64)
#         self.decoder1 = DecoderBlock(64 + 64, 64)
        
#         self.outconv = nn.Sequential(
#             ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(32, num_classes, 1),
#         )
    
#     def forward(self, x, depth, fp=False):

#         # RGB-D fusion
#         e1, e2, e3, e4, e5 = self.rgb_encoder(x)
#         e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
        
#         f5 = self.fusion_block5(e5, e5_d)
#         f4 = self.fusion_block4(e4, e4_d)
#         f3 = self.fusion_block3(e3, e3_d)
#         f2 = self.fusion_block2(e2, e2_d)
#         f1 = self.fusion_block1(e1, e1_d)
        
#         # Decoder
#         d5 = self.decoder5(f5)
#         d4 = self.decoder4(torch.cat([d5, f4], dim=1))
#         d3 = self.decoder3(torch.cat([d4, f3], dim=1))
#         d2 = self.decoder2(torch.cat([d3, f2], dim=1))
#         d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
#         out1 = self.outconv(d1)
#         final_output = F.sigmoid(out1)
#         if fp:
#             return final_output, f5
#         else:
#             return final_output

# class DepthFusion_ResNet34U_f_EMAEncoderOnly1(nn.Module):
#     def __init__(self, num_classes, dropout=0.1):
#         super().__init__()
#         self.rgb_encoder = encoder(num_classes=None)

#         self.depth_encoder = encoder(num_classes=None)
        
#         # Fusion blocks
#         self.fusion_block5 = SEFusionBlock(512, 512, 512)
#         self.fusion_block4 = SEFusionBlock(256, 256, 256)
#         self.fusion_block3 = SEFusionBlock(128, 128, 128)
#         self.fusion_block2 = SEFusionBlock(64, 64, 64)
#         self.fusion_block1 = SEFusionBlock(64, 64, 64)
        
#         self.decoder5 = DecoderBlock(512, 512)
#         self.decoder4 = DecoderBlock(512 + 256, 256)
#         self.decoder3 = DecoderBlock(256 + 128, 128)
#         self.decoder2 = DecoderBlock(128 + 64, 64)
#         self.decoder1 = DecoderBlock(64 + 64, 64)
        
#         self.outconv = nn.Sequential(
#             ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(32, num_classes, 1),
#         )
    
#     def forward(self, x, depth, fp=False):

#         # RGB-D fusion
#         e1, e2, e3, e4, e5 = self.rgb_encoder(x)
#         e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
        
#         f5 = self.fusion_block5(e5, e5_d)
#         f4 = self.fusion_block4(e4, e4_d)
#         f3 = self.fusion_block3(e3, e3_d)
#         f2 = self.fusion_block2(e2, e2_d)
#         f1 = self.fusion_block1(e1, e1_d)
        
#         # Decoder
#         d5 = self.decoder5(f5)
#         d4 = self.decoder4(torch.cat([d5, f4], dim=1))
#         d3 = self.decoder3(torch.cat([d4, f3], dim=1))
#         d2 = self.decoder2(torch.cat([d3, f2], dim=1))
#         d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
#         out1 = self.outconv(d1)
#         final_output = F.sigmoid(out1)
#         if fp:
#             fea = {'rgb_encode': e5, 'depth_encode': e5_d, 'fusion': f5}
#             return final_output, fea
#         else:
#             return final_output


# class ResidualSEFusion(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(c1 + c2, c1, 3, padding=1),
#             nn.BatchNorm2d(c1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x1, x2):
#         concat = torch.cat([x1, x2], dim=1)
#         return self.conv(concat)


# class DepthResidualSEFusion_ResNet34U_f_EMAEncoderOnly(nn.Module):
#     def __init__(self, num_classes, dropout=0.1):
#         super().__init__()
#         self.rgb_encoder = encoder(num_classes=None)

#         self.depth_encoder = encoder(num_classes=None)
        
#         # Fusion blocks
#         self.fusion_block5 = ResidualSEFusion(512, 512)
#         self.fusion_block4 = ResidualSEFusion(256, 256)
#         self.fusion_block3 = ResidualSEFusion(128, 128)
#         self.fusion_block2 = ResidualSEFusion(64, 64)
#         self.fusion_block1 = ResidualSEFusion(64, 64)
        
#         # up_channel_maker = lambda c_in, c_out: nn.Sequential(
#         #     nn.Conv2d(c_in, c_out, kernel_size=1),
#         #     nn.BatchNorm2d(c_out),
#         #     nn.ReLU(inplace=True)
#         # )
#         # self.up_channels = nn.ModuleList([
#         #     up_channel_maker(512, 512),
#         #     up_channel_maker(256, 256),
#         #     up_channel_maker(128, 128),
#         #     up_channel_maker(64, 64),
#         #     up_channel_maker(64, 64),
#         # ])

#         self.decoder5 = DecoderBlock(512, 512)
#         self.decoder4 = DecoderBlock(512 + 256, 256)
#         self.decoder3 = DecoderBlock(256 + 128, 128)
#         self.decoder2 = DecoderBlock(128 + 64, 64)
#         self.decoder1 = DecoderBlock(64 + 64, 64)
        
#         self.outconv = nn.Sequential(
#             ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(32, num_classes, 1),
#         )
    
#     def forward(self, x, depth, fp=False):

#         # RGB-D fusion
#         e1, e2, e3, e4, e5 = self.rgb_encoder(x)
#         e1_d, e2_d, e3_d, e4_d, e5_d = self.depth_encoder(depth)
        
#         res_f5 = self.fusion_block5(e5, e5_d)
#         res_f4 = self.fusion_block4(e4, e4_d)
#         res_f3 = self.fusion_block3(e3, e3_d)
#         res_f2 = self.fusion_block2(e2, e2_d)
#         res_f1 = self.fusion_block1(e1, e1_d)
        
#         # f5 = res_f5 + self.up_channels[0](e5)
#         # f4 = res_f4 + self.up_channels[1](e4)
#         # f3 = res_f3 + self.up_channels[2](e3)
#         # f2 = res_f2 + self.up_channels[3](e2)
#         # f1 = res_f1 + self.up_channels[4](e1)
#         f5 = res_f5 + e5
#         f4 = res_f4 + e4
#         f3 = res_f3 + e3
#         f2 = res_f2 + e2
#         f1 = res_f1 + e1
        
        
#         # Decoder
#         d5 = self.decoder5(f5)
#         d4 = self.decoder4(torch.cat([d5, f4], dim=1))
#         d3 = self.decoder3(torch.cat([d4, f3], dim=1))
#         d2 = self.decoder2(torch.cat([d3, f2], dim=1))
#         d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        
#         out1 = self.outconv(d1)
#         final_output = F.sigmoid(out1)
#         if fp:
#             fea = {'rgb_encode': e5, 'depth_encode': e5_d, 'fusion': f5, 'res_fusion': res_f5}
#             return final_output, fea
#         else:
#             return final_output

class ResNet34U_f_ExtendDAv2(nn.Module):
    """
    Teacher with a frozen Depth Anything V2 backbone (ViT-S or ViT-B) instead of a ResNet-34 depth encoder.

    Key design decisions:
    - DAv2 encoder is fully frozen; only projection layers + fusion blocks + decoder are trained.
    - Four ViT layers are projected to ResNet-34 stage widths and fused at e2..e5 (e1 skipped).
    - Student at inference is RGB-only; DAv2 is training-time only on the teacher.
    """
    dav2_choice = {
        "depth-anything/Depth-Anything-V2-Small-hf": {
            "dav2_dim": 384,
            "layer_indices": [3, 6, 9, 12],
            "proj_channels": [512, 256, 128, 64]
        },
        "depth-anything/Depth-Anything-V2-Base-hf": {
            "dav2_dim": 768,
            "layer_indices": [3, 6, 9, 12],
            "proj_channels": [512, 256, 128, 64]
        }
    }
    def __init__(self, num_classes, dropout=0.1, dav2_model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        
        if dav2_model_name not in self.dav2_choice:
            allowed = ", ".join(sorted(self.dav2_choice.keys()))
            raise ValueError(
                f"Unknown dav2_model_name={dav2_model_name!r}. "
                f"Supported ids: {allowed}"
            )
        
        # RGB encoder — EMA-updated during SSL training (same as current architecture)
        self.rgb_encoder = encoder(num_classes=None)
        # Frozen DAv2 backbone (privileged geometric features)
        
        dav2 = AutoModelForDepthEstimation.from_pretrained(dav2_model_name)
        self.dav2_encoder = dav2.backbone 
        for p in self.dav2_encoder.parameters():
            p.requires_grad = False
        
        # Patch tokens → 1x1 conv to ResNet stage channels; spatial match via interpolate in forward().
        self.dav2_dim = self.dav2_choice[dav2_model_name]['dav2_dim']
        self.dav2_layer_indices = self.dav2_choice[dav2_model_name]['layer_indices']
        self.proj_channels = self.dav2_choice[dav2_model_name]['proj_channels']
        self.proj5 = self._make_proj(self.dav2_dim, self.proj_channels[0])  # layer 12 → matches e5 (512ch, 10x10)
        self.proj4 = self._make_proj(self.dav2_dim, self.proj_channels[1])  # layer  9 → matches e4 (256ch, 20x20)
        self.proj3 = self._make_proj(self.dav2_dim, self.proj_channels[2])  # layer  6 → matches e3 (128ch, 40x40)
        self.proj2 = self._make_proj(self.dav2_dim, self.proj_channels[3])   # layer  3 → matches e2 ( 64ch, 80x80)
        
        # SE-guided fusion blocks — same as DepthFusion_ResNet34U_f_EMAEncoderOnly
        self.fusion_block5 = SEFusionBlock(512, self.proj_channels[0], 512)
        self.fusion_block4 = SEFusionBlock(256, self.proj_channels[1], 256)
        self.fusion_block3 = SEFusionBlock(128, self.proj_channels[2], 128)
        self.fusion_block2 = SEFusionBlock(64,  self.proj_channels[3],  64)
        
        # Standard U-Net decoder with skip connections
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512 + 256, 256)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        self.decoder2 = DecoderBlock(128 + 64,  64)
        self.decoder1 = DecoderBlock(64  + 64,  64)  # e1 passed directly, no DAv2 fusion
        
        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )
    
    def _make_proj(self, in_dim, out_channels):
        """
        1x1 conv projection: maps DAv2 channel dim to ResNet-34 channel dim at a given level.
        Spatial resizing is handled separately via F.interpolate in forward().
        """
        return nn.Sequential(
            nn.Conv2d(in_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _extract_dav2_features(self, x):
        """
        Four intermediate backbone hidden states as spatial maps (patch/14 grid; e.g. 22×22 for 320×320).
        CLS dropped; channel width is ``self.dav2_dim`` (384 Small, 768 Base).
        """
        with torch.no_grad():
            hidden_states = self.dav2_encoder(
                pixel_values=x,
                output_hidden_states=True
            ).hidden_states
        
        feats = []
        for idx in self.dav2_layer_indices:
            h = hidden_states[idx]
            h = h[:, 1:, :]             # drop CLS → [B, N, D]
            B, N, D = h.shape
            h = h.permute(0, 2, 1)      # [B, D, N]
            side = int(np.sqrt(N))
            h = h.reshape(B, D, side, side)
            feats.append(h)
        
        # feats[0] = layer 3  (low-level, closer to edges/textures)
        # feats[3] = layer 12 (high-level, semantic/geometric)
        return feats
    
    def forward(self, x, fp=False, feature_layers=1, type='mixed'):
        # --- RGB encoder ---
        e1, e2, e3, e4, e5 = self.rgb_encoder(x)
        # e1: [B,  64, 160, 160]
        # e2: [B,  64,  80,  80]
        # e3: [B, 128,  40,  40]
        # e4: [B, 256,  20,  20]
        # e5: [B, 512,  10,  10]

        # --- DAv2 feature extraction (no gradient) ---
        dav2_feats = self._extract_dav2_features(x)
        # each: [B, dav2_dim, H', W'] with H',W' from patch size (e.g. 22×22 @ 320, patch 14)

        # --- Project + bilinear resize to match ResNet spatial dims ---
        def proj_and_resize(proj_layer, feat, target_feat):
            out = proj_layer(feat)  # [B, out_c, 22, 22]
            return F.interpolate(
                out, size=target_feat.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        d2 = proj_and_resize(self.proj2, dav2_feats[0], e2)  # [B,  64,  80,  80]
        d3 = proj_and_resize(self.proj3, dav2_feats[1], e3)  # [B, 128,  40,  40]
        d4 = proj_and_resize(self.proj4, dav2_feats[2], e4)  # [B, 256,  20,  20]
        d5 = proj_and_resize(self.proj5, dav2_feats[3], e5)  # [B, 512,  10,  10]

        # --- SE-guided fusion at each encoder level ---
        f5 = self.fusion_block5(e5, d5)
        f4 = self.fusion_block4(e4, d4)
        f3 = self.fusion_block3(e3, d3)
        f2 = self.fusion_block2(e2, d2)
        # e1 has no DAv2 counterpart — passed directly to decoder

        # --- U-Net decoder with skip connections ---
        dec5 = self.decoder5(f5)
        dec4 = self.decoder4(torch.cat([dec5, f4], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, f3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, f2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, e1], dim=1))

        decoder_fea_layers = [dec1, dec2, dec3, dec4, dec5]
        rgb_encoder_fea_layers = [e1,e2,e3,e4,e5]
        mix_encoder_fea_layers = [e1, f2, f3, f4, f5]
        out = self.outconv(dec1)
        final_output = torch.sigmoid(out)

        if fp:
            if type == 'decoder':
                return final_output, decoder_fea_layers[feature_layers - 1]
            elif type == 'rgb_encoder':
                return final_output, rgb_encoder_fea_layers[feature_layers - 1]
            elif type == 'mix_encoder':
                return final_output, mix_encoder_fea_layers[feature_layers - 1]
            else:
                raise ValueError(f"Invalid type: {type}, allowed types are 'decoder', 'rgb_encoder', 'mix_encoder'")

        return final_output


class MaxMeanOperation(nn.Module):
    """Max + mean over channel dim → 2 maps"""

    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)),
            dim=1,
        )


def _bf_conv(inp_dim, out_dim, kernel_size=3, bn=False, relu=True, bias=True):
    layers = [
        nn.Conv2d(
            inp_dim,
            out_dim,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_dim))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class BiFusionResidual(nn.Module):
    """Fuses concat[g', x', bp] → ch_out (TransFuse-style bottleneck + skip)."""

    def __init__(self, inp_dim, out_dim):
        super().__init__()
        mid = out_dim // 2
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = nn.Conv2d(inp_dim, mid, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_dim, kernel_size=1, bias=False)
        self.need_skip = inp_dim != out_dim
        self.skip_layer = nn.Conv2d(inp_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.skip_layer(x) if self.need_skip else x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out + residual


class BiFusionBlock(nn.Module):
    """
    TransFuse-style fusion: ``g`` = CNN (local), ``x`` = transformer/DAv2 (same H×W).

    - ``ch_1``, ``ch_2``: input channel widths (may differ). ``W_g`` / ``W_x`` map both to ``ch_int``
      for the bilinear term ``W_g(g) * W_x(x)``.
    - Spatial attention on ``g``; channel attention (squeeze on ``x``) on ``x``.
    - ``ch_out``: fused output width (match CNN stage for U-Net decoder).
    - ``r_2``: bottleneck divisor for channel MLP on ``x`` (larger → narrower bottleneck).
    """

    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.0):
        super().__init__()
        hidden2 = max(ch_2 // r_2, 1)
        self.fc1 = nn.Conv2d(ch_2, hidden2, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.compress = MaxMeanOperation()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.W_g = nn.Sequential(
            nn.Conv2d(ch_1, ch_int, kernel_size=1, bias=False), 
            nn.BatchNorm2d(ch_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(ch_2, ch_int, kernel_size=1, bias=False), 
            nn.BatchNorm2d(ch_int)
        )
        
        self.W = nn.Sequential(
            nn.Conv2d(ch_int, ch_int, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(ch_int), 
            nn.ReLU(inplace=True)
        )


        self.relu = nn.ReLU(inplace=True)
        self.residual = BiFusionResidual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate) if drop_rate > 0 else None

    def forward(self, g, x):
        bp = self.W(self.W_g(g) * self.W_x(x))

        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in

        fuse = self.residual(torch.cat([g, x, bp], dim=1))
        if self.dropout is not None:
            return self.dropout(fuse)
        return fuse


class ResNet34U_f_ExtendDAv2_1(nn.Module):
    """
    DAv2 teacher: learned pre/post conv around bilinear resize (DAv2 stays ``dav2_dim`` channels).
    BiFusion fuses CNN ``e*`` with aligned DAv2 maps (``ch_1`` ≠ ``ch_2``). Optional ``+ e_k`` residual.
    """
    dav2_choice = {
        "depth-anything/Depth-Anything-V2-Small-hf": {
            "dav2_dim": 384,
            "layer_indices": [3, 6, 9, 12],
            "proj_channels": [512, 256, 128, 64]
        },
        "depth-anything/Depth-Anything-V2-Base-hf": {
            "dav2_dim": 768,
            "layer_indices": [3, 6, 9, 12],
            "proj_channels": [512, 256, 128, 64]
        }
    }
    def __init__(
        self,
        num_classes,
        dropout=0.1,
        dav2_model_name="depth-anything/Depth-Anything-V2-Small-hf",
        use_cnn_residual=True,
        bifusion_drop=0.0,
    ):
        super().__init__()
        
        if dav2_model_name not in self.dav2_choice:
            allowed = ", ".join(sorted(self.dav2_choice.keys()))
            raise ValueError(
                f"Unknown dav2_model_name={dav2_model_name!r}. "
                f"Supported ids: {allowed}"
            )
        
        self.use_cnn_residual = use_cnn_residual

        # RGB encoder — EMA-updated during SSL training (same as current architecture)
        self.rgb_encoder = encoder(num_classes=None)
        # Frozen DAv2 backbone (privileged geometric features)
        
        dav2 = AutoModelForDepthEstimation.from_pretrained(dav2_model_name)
        self.dav2_encoder = dav2.backbone 
        for p in self.dav2_encoder.parameters():
            p.requires_grad = False
        
        self.dav2_dim = self.dav2_choice[dav2_model_name]['dav2_dim']
        self.dav2_layer_indices = self.dav2_choice[dav2_model_name]['layer_indices']


        dv2 = self.dav2_dim
        self.pre_align = nn.ModuleList(
            [
                self._make_pre_align(dv2, dv2),
                self._make_pre_align(dv2, dv2),
                self._make_pre_align(dv2, dv2),
                self._make_pre_align(dv2, dv2),
            ]
        )
        self.post_align = nn.ModuleList(
            [
                self._make_post_align(dv2, dv2),
                self._make_post_align(dv2, dv2),
                self._make_post_align(dv2, dv2),
                self._make_post_align(dv2, dv2)
            ]
        )
        # ch_int: bilinear interaction width; r_2: channel-SE bottleneck on transformer branch
        self.fusion_block5 = BiFusionBlock(512, dv2, r_2=16, ch_int=256, ch_out=512, drop_rate=bifusion_drop)
        self.fusion_block4 = BiFusionBlock(256, dv2, r_2=16, ch_int=128, ch_out=256, drop_rate=bifusion_drop)
        self.fusion_block3 = BiFusionBlock(128, dv2, r_2=16, ch_int=64, ch_out=128, drop_rate=bifusion_drop)
        self.fusion_block2 = BiFusionBlock(64, dv2, r_2=8, ch_int=32, ch_out=64, drop_rate=bifusion_drop)
        
        # Standard U-Net decoder with skip connections
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512 + 256, 256)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        self.decoder2 = DecoderBlock(128 + 64,  64)
        self.decoder1 = DecoderBlock(64  + 64,  64)  # e1 passed directly, no DAv2 fusion
        
        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )
    
    def _make_pre_align(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            ConvBlock(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        )
    def _make_post_align(self, in_dim, out_dim):
        return ConvBlock(in_dim, out_dim, kernel_size=3, stride=1, padding=1)   

    def _extract_dav2_features(self, x):
        """
        Four intermediate backbone hidden states as spatial maps (patch/14 grid; e.g. 22×22 for 320×320).
        CLS dropped; channel width is ``self.dav2_dim`` (384 Small, 768 Base).
        """
        with torch.no_grad():
            hidden_states = self.dav2_encoder(
                pixel_values=x,
                output_hidden_states=True
            ).hidden_states
        
        feats = []
        for idx in self.dav2_layer_indices:
            h = hidden_states[idx]
            h = h[:, 1:, :]             # drop CLS → [B, N, D]
            B, N, D = h.shape
            h = h.permute(0, 2, 1)      # [B, D, N]
            side = int(np.sqrt(N))
            h = h.reshape(B, D, side, side)
            feats.append(h)
        
        # feats[0] = layer 3  (low-level, closer to edges/textures)
        # feats[3] = layer 12 (high-level, semantic/geometric)
        return feats
    
    def forward(self, x, fp=False, feature_layers=1, type='mixed'):
        # --- RGB encoder ---
        e1, e2, e3, e4, e5 = self.rgb_encoder(x)
        # e1: [B,  64, 160, 160]
        # e2: [B,  64,  80,  80]
        # e3: [B, 128,  40,  40]
        # e4: [B, 256,  20,  20]
        # e5: [B, 512,  10,  10]

        # --- DAv2 feature extraction (no gradient) ---
        dav2_feats = self._extract_dav2_features(x)
        # each: [B, dav2_dim, H', W'] with H',W' from patch size (e.g. 22×22 @ 320, patch 14)

        # --- Align DAv2 maps to each CNN stage: pre (patch grid) → resize → post (CNN resolution) ---
        def align_stage(stage_idx, feat, target_spatial):
            y = self.pre_align[stage_idx](feat)
            y = F.interpolate(y, size=target_spatial, mode="bilinear", align_corners=False)
            return self.post_align[stage_idx](y)

        d2 = align_stage(0, dav2_feats[0], e2.shape[2:])
        d3 = align_stage(1, dav2_feats[1], e3.shape[2:])
        d4 = align_stage(2, dav2_feats[2], e4.shape[2:])
        d5 = align_stage(3, dav2_feats[3], e5.shape[2:])

        # --- BiFusion: g = CNN, x = DAv2 (same H×W; channels differ) ---
        f5 = self.fusion_block5(e5, d5)
        f4 = self.fusion_block4(e4, d4)
        f3 = self.fusion_block3(e3, d3)
        f2 = self.fusion_block2(e2, d2)
        if self.use_cnn_residual:
            f5 = f5 + e5
            f4 = f4 + e4
            f3 = f3 + e3
            f2 = f2 + e2
        # e1 has no DAv2 counterpart — passed directly to decoder

        # --- U-Net decoder with skip connections ---
        dec5 = self.decoder5(f5)
        dec4 = self.decoder4(torch.cat([dec5, f4], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, f3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, f2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, e1], dim=1))

        decoder_fea_layers = [dec1, dec2, dec3, dec4, dec5]
        rgb_encoder_fea_layers = [e1,e2,e3,e4,e5]
        mix_encoder_fea_layers = [e1, f2, f3, f4, f5]
        out = self.outconv(dec1)
        final_output = torch.sigmoid(out)

        if fp:
            if type == 'decoder':
                return final_output, decoder_fea_layers[feature_layers - 1]
            elif type == 'rgb_encoder':
                return final_output, rgb_encoder_fea_layers[feature_layers - 1]
            elif type == 'mix_encoder':
                return final_output, mix_encoder_fea_layers[feature_layers - 1]
            else:
                raise ValueError(f"Invalid type: {type}, allowed types are 'decoder', 'rgb_encoder', 'mix_encoder'")

        return final_output
        

# if __name__ == "__main__":
#     # rgb = torch.randn(1, 3, 320, 320)
#     # depth = torch.randn(1, 3, 320, 320)
#     # mask = torch.randn(1, 2, 320, 320)