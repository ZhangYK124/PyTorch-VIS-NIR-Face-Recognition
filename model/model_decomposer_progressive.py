'''
star GAN + disentangled learning
progressive learning + no patches

We employ a latent domain discriminator to obtain the common feature space between domains rather than MMD.

In order to obtain a high-resolution generative face, we may first generate a larger face images than its original size and down-sample, e.g. 224x224 --> 168x168 --> 112x112.

Three branches might work. Check it out!

w/o CAM

Reference:
Face Sketch Synthesis by Multi-domain Adversarial Learning.
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
    def __init__(self, ngf=64, n_blocks=3):
        super(Style_Encoder, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cls = nn.Linear(128, 3)
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
        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2 ** i
            Style += [nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]
        
        mult = 2 * n_downsampling
        for i in range(n_blocks):
            Style += [_Residual_Block(ngf * mult)]
        
        self.Style = nn.Sequential(*Style)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.Style(x)
        style_feature = F.adaptive_avg_pool2d(x, 1)
        style_logit = self.cls(style_feature.view(style_feature.shape[0], -1))
        style_feature = self.deconv(style_feature)
        return style_feature, style_logit

class Encoder(nn.Module):
    '''
    Style + Intrinsic
    '''
    def __init__(self,ngf=64,n_blocks=3):
        super(Encoder,self).__init__()
        # **************************    Input    *************************
        Input = []
        Input += [nn.Conv2d(in_channels=3,out_channels=ngf,kernel_size=3,stride=1,padding=1,bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=0,bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.LeakyReLU(0.1,inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.LeakyReLU(0.1, inplace=True)
                  ]
        self.Input = nn.Sequential(*Input)

        # **************************    Intrinsic    *************************
        Intrinsic = []
        

    def forward(self, x):
        x = self.Input(x)
        return x

class Intrinsic_Encoder(nn.Module):
    def __init__(self, ngf=64, n_blocks=3):
        super(Intrinsic_Encoder, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=3, stride=1, padding=1, bias=False)
        
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
        
        self.Intrinsic = nn.Sequential(*Intrinsic_Decomposer)
    
    def forward(self, x):
        x = self.conv_in(x)
        intrinsic = self.Intrinsic(x)
        return intrinsic


class Integrator(nn.Module):
    def __init__(self, ngf=64, n_blocks=6):
        super(Integrator, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=4, padding=1, output_padding=0),
            nn.ReflectionPad2d(1),
            nn.InstanceNorm2d(128)
        )

        # Class Activation Map
        mult = 2
        self.gap_fc = nn.Linear(ngf * mult * 2, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult * 2, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 4, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Down sampling
        n_downsampling = 1
        # Up sampling
        Model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Model = [nn.Upsample(scale_factor=2, mode='nearest'),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=1, padding=0, bias=False),
                     nn.InstanceNorm2d(ngf * mult / 2),
                     nn.ReLU(True)]
        for i in range(n_blocks):
            Model += [_Residual_Block(ngf)]
        Model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=0, bias=True),
                  nn.Tanh()]
        
        self.model = nn.Sequential(*Model)
    
    def forward(self, intrinsic, style):
        # style = self.deconv(style)
        feature = torch.cat([intrinsic, style], 1)

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
        
        return out,heatmap,cam_logit


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=128, n_blocks=10, img_size=112):
        super(Generator, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual3 = self.make_layer(_Residual_Block, 3, in_channel=ngf, out_channel=ngf, kernel_size=7, padding=3, stride=1)
        self.residual6_2 = self.make_layer(_Residual_Block, 6, in_channel=ngf, out_channel=ngf, kernel_size=7, padding=3, stride=1)
        self.residual = self.make_layer(_Residual_Block, 1, in_channel=192, out_channel=192, kernel_size=7, padding=3, stride=1)
        self.bn = nn.BatchNorm2d(128, affine=True)
        self.bn_out = nn.BatchNorm2d(192, affine=True)
        self.in_out = nn.InstanceNorm2d(192, affine=True)
        self.instance = nn.InstanceNorm2d(128, affine=True)
        self.conv_out = nn.Conv2d(in_channels=192, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.out = nn.Tanh()
        
        Encoder = []
        Encoder += [nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm2d(ngf),
                    nn.ReLU(inplace=True)]
        '''
        # Down sampling
        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2 ** i
            Encoder += [nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                        nn.InstanceNorm2d(ngf * mult * 2),
                        nn.ReLU(True)]
        '''
        
        for i in range(n_blocks):
            Encoder += [_Residual_Block(ngf)]
        
        '''
        # Class Activation Map
        self.gap_fc_vis = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc_vis = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1_vis = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu_vis = nn.ReLU(True)
        
        self.gap_fc_nir = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc_nir = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1_nir = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu_nir = nn.ReLU(True)
        
        self.gap_fc_sketch = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc_sketch = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1_sketch = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu_sketch = nn.ReLU(True)
        '''
        
        Decoder_VIS = []
        for i in range(n_blocks):
            Decoder_VIS += [_Residual_Block(ngf),
                            nn.InstanceNorm2d(ngf)]
        Decoder_VIS += [nn.ReflectionPad2d(1),
                        nn.Conv2d(in_channels=ngf,out_channels=3,kernel_size=3,stride=1,padding=0,bias=False),
                        nn.Tanh()]
        
        Decoder_NIR = []
        for i in range(n_blocks):
            Decoder_NIR += [_Residual_Block(ngf),
                            nn.InstanceNorm2d(ngf)]
        Decoder_NIR += [nn.ReflectionPad2d(1),
                        nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False),
                        nn.Tanh()]
        
        Decoder_SKETCH = []
        for i in range(n_blocks):
            Decoder_SKETCH += [_Residual_Block(ngf),
                               nn.InstanceNorm2d(ngf)]
        Decoder_SKETCH += [nn.ReflectionPad2d(1),
                        nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False),
                        nn.Tanh()]
        
        self.Encoder = nn.Sequential(*Encoder)
        self.Decoder_VIS = nn.Sequential(*Decoder_VIS)
        self.Decoder_NIR = nn.Sequential(*Decoder_NIR)
        self.Decoder_SKETCH = nn.Sequential(*Decoder_SKETCH)
    
    def make_layer(self, block, num_of_layer, in_channel, out_channel, kernel_size, stride, padding):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channel, out_channel, kernel_size, stride, padding))
        return nn.Sequential(*layers)
    
    def forward(self, I112, c):
        x = self.Encoder(I112)
        # gap = torch.nn.functional.adaptive_avg_pool2d(x,1)
        # gap_logit = self.gap_
        if c == 0:
            Decoder = self.Decoder_VIS
        elif c == 1:
            Decoder = self.Decoder_NIR
        elif c == 2:
            Decoder = self.Decoder_SKETCH
        out = Decoder(x)
        return out


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    
    def __init__(self, image_size=112, conv_dim=64, c_dim=3, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01)),
            layers.append(_Residual_Block(curr_dim * 2))
            layers.append(nn.InstanceNorm2d(curr_dim * 2))
            layers.append(_Residual_Block(curr_dim * 2))
            layers.append(nn.InstanceNorm2d(curr_dim * 2))
            curr_dim = curr_dim * 2
        
        kernel_size = int(image_size // np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
    
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
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
    model = Generator().cuda()
    input = Variable(torch.rand(8, 3, 112, 112)).cuda()
    # c = 1
    # writter = SummaryWriter('./log')
    src, c, _, _, _, _ = model(input, 1)
    batch_size = cls.size(0)
    out = torch.zeros(batch_size, 3)
    out[np.arange(batch_size), 2] = 1
    print(model(input)[0].size())
