'''
We downsample the intrinsic feature to channel x 1 x 1 rather than channel x 112 x 112.
'''

import torch
import numpy as np

import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

import pdb


class _Residual_Block(nn.Module):
    def __init__(self, dim, use_bias=False, kernel_size=3, stride=1, padding=1):
        super(_Residual_Block, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(padding),
                       nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1, inplace=True)]
        
        conv_block += [nn.ReflectionPad2d(padding),
                       nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]
        
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes * 2, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes * 2)
        self.relu = nn.PReLU(planes * 2)
        self.conv2 = conv3x3(planes * 2, planes)
        self.bn2 = nn.InstanceNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)
    
    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)
    
    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)
    
    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)
        
        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out
    
    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class Style_Encoder(nn.Module):
    def __init__(self, ngf=64, n_blocks=2):
        super(Style_Encoder, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cls = nn.Linear(256, 3)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.ReflectionPad2d(1),
            nn.InstanceNorm2d(128)
        )
        # **************************    Style    *************************
        Style = []
        Style += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.ReLU(inplace=True)]
        
        # Down sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            Style += [nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]
        
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            Style += [_Residual_Block(ngf * mult)]
        
        self.Style = nn.Sequential(*Style)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.Style(x)
        style_feature = F.adaptive_avg_pool2d(x, 1)
        style_logit = self.cls(style_feature.view(style_feature.shape[0], -1))
        # style_feature = self.deconv(style_feature)
        return style_feature, style_logit


class Intrinsic_Encoder(nn.Module):
    def __init__(self, ngf=64, n_blocks=2):
        super(Intrinsic_Encoder, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=3, stride=1, padding=1, bias=False)
        
        # **************************    Intrinsic    *************************
        Intrinsic_Decomposer = []
        Intrinsic_Decomposer += [nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                                 nn.InstanceNorm2d(ngf * 2),
                                 nn.ReLU(inplace=True)]
        
        # Down sampling
        n_downsampling = 4
        for i in range(n_downsampling):
            mult = 2 ** i
            Intrinsic_Decomposer += [nn.ReflectionPad2d(1),
                                     nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                                     nn.InstanceNorm2d(ngf * mult * 2),
                                     nn.ReLU(True)]
            for j in range(n_blocks):
                Intrinsic_Decomposer += [_Residual_Block(ngf * 2 * mult)]
            Intrinsic_Decomposer += [nn.InstanceNorm2d(ngf * 2 * mult)]
        Intrinsic_Decomposer += [nn.Conv2d(ngf * 2 * mult, ngf * 2 * mult * 2, kernel_size=7, stride=1, padding=0, bias=False)]
        
        # mult = 2 ** n_downsampling
        # for i in range(n_blocks):
        #     Intrinsic_Decomposer += [_Residual_Block(ngf * 2 * mult),
        #                              nn.InstanceNorm2d(ngf * 2 * mult)]
        
        self.Intrinsic = nn.Sequential(*Intrinsic_Decomposer)
    
    def make_layer(self, block, num_of_layer, in_channel, out_channel, kernel_size, stride, padding):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channel, out_channel, kernel_size, stride, padding))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_in(x)
        intrinsic = self.Intrinsic(x)
        return intrinsic


class Integrator(nn.Module):
    def __init__(self, ngf=64, n_blocks=10):
        super(Integrator, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2304, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.ReflectionPad2d(1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=2, output_padding=0),
            nn.ReflectionPad2d(1),
            nn.InstanceNorm2d(128)
        )
        
        # Class Activation Map
        mult = 1
        self.gap_fc = nn.Linear(ngf * mult * 2, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult * 2, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 4, ngf * mult * 2, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Down sampling
        n_downsampling = 1
        # Up sampling
        Model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Model += [
                # nn.ConvTranspose2d(in_channels=ngf * mult, out_channels=ngf*mult,kernel_size=3,stride=2,padding=1,bias=False,output_padding=1),
                # nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.ReflectionPad2d(1),
                nn.Conv2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult / 2),
                nn.ReLU(True)]
        for i in range(n_blocks):
            Model += [_Residual_Block(ngf)]
            Model += [nn.InstanceNorm2d(ngf)]
        
        Model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=0, bias=True),
                  nn.Tanh()]
        
        self.model = nn.Sequential(*Model)
    
    def make_layer(self, block, num_of_layer, in_channel, out_channel, kernel_size, stride, padding):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channel, out_channel, kernel_size, stride, padding))
        return nn.Sequential(*layers)
    
    def forward(self, intrinsic, style):
        feature = torch.cat([intrinsic, style], 1)
        feature = self.deconv(feature)
        # feature = torch.cat([intrinsic, style_feature], 1)
        
        gap = F.adaptive_avg_pool2d(feature, 1)
        gap_logit = self.gap_fc(gap.view(feature.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = feature * gap_weight.unsqueeze(2).unsqueeze(3)
        
        gmp = F.adaptive_max_pool2d(feature, 1)
        gmp_logit = self.gmp_fc(gmp.view(feature.shape[0], -1))
        # gmp_logit = self.softmax(gmp_logit)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = feature * gmp_weight.unsqueeze(2).unsqueeze(3)
        
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        # out_cls = (gap_logit+gmp_logit)/2.0
        feature = torch.cat([gap, gmp], 1)
        feature = self.relu(self.conv1x1(feature))
        
        heatmap = torch.sum(feature, dim=1, keepdim=True)
        
        out = self.model(feature)
        
        return out, heatmap, cam_logit


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=3, img_size=112):
        super(Generator, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual3 = self.make_layer(_Residual_Block, 3, in_channel=ngf, out_channel=ngf, kernel_size=7, padding=3, stride=1)
        self.residual6_2 = self.make_layer(_Residual_Block, 6, in_channel=128, out_channel=128, kernel_size=7, padding=3, stride=1)
        self.residual = self.make_layer(_Residual_Block, 1, in_channel=192, out_channel=192, kernel_size=7, padding=3, stride=1)
        self.bn = nn.BatchNorm2d(128, affine=True)
        self.bn_out = nn.BatchNorm2d(192, affine=True)
        self.in_out = nn.InstanceNorm2d(192, affine=True)
        self.instance = nn.InstanceNorm2d(128, affine=True)
        self.conv_out = nn.Conv2d(in_channels=192, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.out = nn.Tanh()
        self.conv_cls = nn.Conv2d(ngf * 2, 3, kernel_size=int(112 / np.power(2, 6)), bias=False)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.weight_avg_pooling = nn.AdaptiveAvgPool2d((7, 7))
        self.pad = nn.ReflectionPad2d(3)
        self.softmax = nn.Softmax()
        self.style_cls = nn.Linear(128, 3)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.ReflectionPad2d(1),
            nn.InstanceNorm2d(128)
        )
        
        # **************************    Intrinsic    *************************
        Intrinsic_Decomposer = []
        Intrinsic_Decomposer += [nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                                 nn.InstanceNorm2d(ngf),
                                 nn.ReLU(inplace=True)]
        
        # Down sampling
        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2 ** i
            Intrinsic_Decomposer += [nn.ReflectionPad2d(1),
                                     nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                                     nn.InstanceNorm2d(ngf * mult * 2),
                                     nn.ReLU(True)]
        
        mult = 2 * n_downsampling
        for i in range(n_blocks):
            Intrinsic_Decomposer += [_Residual_Block(ngf * mult)]
        
        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult * 2, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult * 2, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 4, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # **************************    Style    *************************
        Style_Decomposer = []
        Style_Decomposer += [nn.ReflectionPad2d(3),
                             nn.Conv2d(ngf, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                             nn.InstanceNorm2d(ngf),
                             nn.ReLU(inplace=True)]
        
        # Down sampling
        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2 ** i
            Style_Decomposer += [nn.ReflectionPad2d(1),
                                 nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                                 nn.InstanceNorm2d(ngf * mult * 2),
                                 nn.ReLU(True)]
        
        mult = 2 * n_downsampling
        for i in range(n_blocks):
            Style_Decomposer += [_Residual_Block(ngf * mult)]
        
        # Up sampling
        Integrator = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Integrator = [nn.Upsample(scale_factor=2, mode='nearest'),
                          nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=1, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult / 2),
                          nn.ReLU(True)]
        for i in range(n_blocks):
            Integrator += [_Residual_Block(ngf)]
        Integrator += [nn.ReflectionPad2d(1),
                       nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=0, bias=False),
                       nn.InstanceNorm2d(ngf),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=0, bias=True),
                       nn.Tanh()]
        
        self.Intrinsic_Decomposer = nn.Sequential(*Intrinsic_Decomposer)
        self.Style_Decomposer = nn.Sequential(*Style_Decomposer)
        self.Integrator = nn.Sequential(*Integrator)
    
    def make_layer(self, block, num_of_layer, in_channel, out_channel, kernel_size, stride, padding):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channel, out_channel, kernel_size, stride, padding))
        return nn.Sequential(*layers)
    
    def forward(self, x, intrinsic=None, style=None):
        if intrinsic is None:
            x = self.conv_in(x)
            intrinsic_feature = self.Intrinsic_Decomposer(x)
            style_weight = self.Style_Decomposer(x)
            # intrinsic_feature = self.hg(intrinsic_feature)
        
        else:
            intrinsic_feature = intrinsic
            style_weight = style
        
        # index = np.arange(style_weight.shape[0])
        # np.random.shuffle(index)
        # style_weight = style_weight[index]
        # feature = torch.cat([intrinsic_feature,style_weight],1)
        
        # feature = intrinsic_feature.mul(style_weight)
        
        # intrinsic_feature1 = self.pad(intrinsic_feature)
        # style_weight = self.weight_avg_pooling(style_weight)
        # # style_weight = torch.mean(style_weight,dim=0)
        # style_weight = style_weight.expand(128,style_weight.size(0),style_weight.size(1),style_weight.size(2))
        # feature = F.conv2d(intrinsic_feature1,style_weight,stride=1)
        
        # out_cls = self.conv_cls(feature)
        
        style_feature = F.adaptive_avg_pool2d(style_weight, 1)
        style_logit = self.style_cls(style_feature.view(style_feature.shape[0], -1))
        style_feature = self.deconv(style_feature)
        
        feature = torch.cat([intrinsic_feature, style_feature], 1)
        
        gap = F.adaptive_avg_pool2d(feature, 1)
        gap_logit = self.gap_fc(gap.view(feature.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = feature * gap_weight.unsqueeze(2).unsqueeze(3)
        
        gmp = F.adaptive_max_pool2d(feature, 1)
        gmp_logit = self.gmp_fc(gmp.view(feature.shape[0], -1))
        # gmp_logit = self.softmax(gmp_logit)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = feature * gmp_weight.unsqueeze(2).unsqueeze(3)
        
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        # out_cls = (gap_logit+gmp_logit)/2.0
        feature = torch.cat([gap, gmp], 1)
        feature = self.relu(self.conv1x1(feature))
        
        heatmap = torch.sum(feature, dim=1, keepdim=True)
        
        # out_cls = self.avg_pooling(out_cls)
        out = self.Integrator(feature)
        return out, intrinsic_feature, style_weight, heatmap, cam_logit, style_logit


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    
    def __init__(self, image_size=112, conv_dim=128, c_dim=3, repeat_num=2):
        super(Discriminator, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(4096, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.ReflectionPad2d(1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=2, output_padding=0),
            nn.ReflectionPad2d(1),
            nn.InstanceNorm2d(128)
        )
        
        layers = []
        layers.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        
        fc = []
        fc.append(nn.Linear(4096,1024))
        fc.append(nn.Linear(1024,512))
        fc.append(nn.Linear(512,128))
        fc.append(nn.Linear(128,1))
        
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.fc = nn.Sequential(*fc)
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.deconv(x)
        # h = self.main(x)
        out_src = self.fc(x.view(x.shape[0], -1))
        # out_src = self.conv1(h)
        # out_cls = self.conv2(h)
        # out_src = F.adaptive_avg_pool2d(out_src, out_src.size()[2:]).view(out_src.size()[0],-1)
        # return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src


class Discriminator_CAM(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=4):
        super(Discriminator_CAM, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]
        
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]
        
        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        
        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        x = self.model(input)
        
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))
        
        heatmap = torch.sum(x, dim=1, keepdim=True)
        
        x = self.pad(x)
        out = self.conv(x)
        out = F.avg_pool2d(out, out.size()[2:]).view(out.size()[0], -1)
        return out, cam_logit, heatmap


if __name__ == '__main__':
    # model = Discriminator().cuda()
    # model = Generator().cuda()
    model = Integrator().cuda()
    input = Variable(torch.rand(8, 3, 112, 112)).cuda()
    intrinsic = Variable(torch.rand(8,2048,1,1)).cuda()
    style = Variable(torch.rand(8,256,1,1)).cuda()
    # c = 1
    # writter = SummaryWriter('./log')
    src,_,_ = model(intrinsic,style)
    batch_size = cls.size(0)
    out = torch.zeros(batch_size, 3)
    out[np.arange(batch_size), 2] = 1
    print(model(input)[0].size())
