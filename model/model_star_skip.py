'''
star GAN
skip connection + no patches

We employ a latent domain discriminator to obtain the common feature space between domains rather than MMD.

In order to obtain a high-resolution generative face, we may first generate a larger face images than its original size and down-sample, e.g. 224x224 --> 168x168 --> 112x112.


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
    def __init__(self, dim, use_bias=False,kernel_size=3,stride=1,padding=1):
        super(_Residual_Block, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(padding),
                       nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1,inplace=True)]

        conv_block += [nn.ReflectionPad2d(padding),
                       nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Channel Attention Layer
class ChannelAttentionLayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(ChannelAttentionLayer,self).__init__()
        # global average pooling: feature --> points
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_avg = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel,channel//reduction,stride=1,kernel_size=3,padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction,channel,stride=1,kernel_size=3,padding=0,bias=True),
            nn.Sigmoid()
        )
        
        # global max pooling
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv_max = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel,channel//reduction,stride=1,kernel_size=3,padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction,channel,stride=1,kernel_size=3,padding=0,bias=True),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        gap = self.avg_pooling(x).view(x.shape[0],-1)
        y = self.conv_avg(y)
        return x * y
        
        
    
class Generator(nn.Module):
    def __init__(self,input_nc=3, output_nc = 3, ngf=64, n_blocks=6,img_size=112):
        super(Generator,self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False)
        self.residual3 = self.make_layer(_Residual_Block, 3, in_channel=ngf, out_channel=ngf,kernel_size=7,padding=3,stride=1)
        self.residual6_2 = self.make_layer(_Residual_Block, 6, in_channel=128, out_channel=128,kernel_size=7,padding=3,stride=1)
        self.residual = self.make_layer(_Residual_Block, 1, in_channel=192, out_channel=192,kernel_size=7,padding=3,stride=1)
        self.bn = nn.BatchNorm2d(128,affine=True)
        self.bn_out = nn.BatchNorm2d(192, affine=True)
        self.in_out = nn.InstanceNorm2d(192, affine=True)
        self.instance = nn.InstanceNorm2d(128,affine=True)
        self.conv_out = nn.Conv2d(in_channels=192,out_channels=3,kernel_size=3,stride=1,padding=1,bias=False)
        self.out = nn.Tanh()
        
        Encoder = []
        Encoder += [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc,ngf,kernel_size=7,stride=1,padding=0,bias=False),
                    nn.InstanceNorm2d(ngf),
                    nn.ReLU(inplace=True)]
        
        # Down sampling
        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2**i
            Encoder += [nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf*mult, ngf*mult *2, kernel_size=3,stride=2,padding=0,bias=False),
                        nn.InstanceNorm2d(ngf*mult*2),
                        nn.ReLU(True)]
            
        mult = 2*n_downsampling
        for i in range(n_blocks):
            Encoder += [_Residual_Block(ngf*mult)]
        
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
        
        
        
        # Up sampling
        Decoder_VIS = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            Decoder_VIS = [nn.Upsample(scale_factor=2,mode='nearest'),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(ngf*mult,ngf*mult//2,kernel_size=3,stride=1,padding=0,bias=False),
                       nn.InstanceNorm2d(ngf*mult/2),
                       nn.ReLU(True)]
        for i in range(n_blocks):
            Decoder_VIS += [_Residual_Block(ngf)]
        Decoder_VIS += [nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=0,bias=False),
                        nn.InstanceNorm2d(ngf),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf,3,kernel_size=3,stride=1,padding=0,bias=True),
                        nn.Tanh()]

        Decoder_NIR = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Decoder_NIR = [nn.Upsample(scale_factor=2, mode='nearest'),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=1, padding=0, bias=False),
                       nn.InstanceNorm2d(ngf * mult / 2),
                       nn.ReLU(True)]
        for i in range(n_blocks):
            Decoder_NIR += [_Residual_Block(ngf)]
        Decoder_NIR += [nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=0, bias=False),
                        nn.InstanceNorm2d(ngf),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf,3,kernel_size=3,stride=1,padding=0,bias=True),
                        nn.Tanh()]

        Decoder_SKETCH = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Decoder_SKETCH = [nn.Upsample(scale_factor=2, mode='nearest'),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=1, padding=0, bias=False),
                       nn.InstanceNorm2d(ngf * mult / 2),
                       nn.ReLU(True)]
        for i in range(n_blocks):
            Decoder_SKETCH += [_Residual_Block(ngf)]
        Decoder_SKETCH += [nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=0, bias=False),
                        nn.InstanceNorm2d(ngf),
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0, bias=True),
                        nn.Tanh()]
        
        self.Encoder = nn.Sequential(*Encoder)
        self.Decoder_VIS = nn.Sequential(*Decoder_VIS)
        self.Decoder_NIR = nn.Sequential(*Decoder_NIR)
        self.Decoder_SKETCH = nn.Sequential(*Decoder_SKETCH)
        
    def make_layer(self, block, num_of_layer,in_channel, out_channel,kernel_size,stride,padding):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channel,out_channel,kernel_size,stride,padding))
        return nn.Sequential(*layers)
        
    def forward(self, I112,c):
        x = self.Encoder(I112)
        # gap = torch.nn.functional.adaptive_avg_pool2d(x,1)
        # gap_logit = self.gap_
        if c==0:
            Decoder = self.Decoder_VIS
        elif c==1:
            Decoder = self.Decoder_NIR
        elif c==2:
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
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        out_src = F.adaptive_avg_pool2d(out_src, out_src.size()[2:]).view(out_src.size()[0],-1)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
    
if __name__ == '__main__':
    model = Discriminator().cuda()
    # model = Generator().cuda()
    input = Variable(torch.rand(8,3,112,112)).cuda()
    # c = 1
    # writter = SummaryWriter('./log')
    src,cls = model(input)
    batch_size = cls.size(0)
    out = torch.zeros(batch_size,3)
    out[np.arange(batch_size),2] = 1
    print(model(input)[0].size())
        
        
