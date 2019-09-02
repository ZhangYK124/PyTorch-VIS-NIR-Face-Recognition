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

class LocalPathWay(nn.Module):
    def __init__(self):
        super(LocalPathWay, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual = self.make_layer(_Residual_Block, 3, dim=64)
        self.bn = nn.BatchNorm2d(64, affine=True)
        self.instance = nn.InstanceNorm2d(64, affine=True)
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.out = nn.Tanh()
    
    def make_layer(self, block, num_of_layer, dim):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(dim=dim))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.residual(out)
        out = self.residual(out)
        out_feature = self.instance(out)
        local_img = self.conv_out(out_feature)
        local_img = self.out(local_img)
        return out_feature, local_img
        # return local_img


class LocalFuser(nn.Module):
    '''
    Landmark coordinate:
    x       y

    '''
    def __init__(self):
        super(LocalFuser, self).__init__()
    
    def forward(self, f_left_eye, f_right_eye):
        EYE_W, EYE_H = 22, 22
        IMG_SIZE = 112
        f_left_eye_out = torch.nn.functional.pad(f_left_eye, (38 - EYE_W // 2 - 1, IMG_SIZE - (38 + EYE_W // 2 - 1), 54 - EYE_H // 2 - 1, IMG_SIZE - (54 + EYE_H // 2 - 1)), 'constant', 0)
        f_right_eye_out = torch.nn.functional.pad(f_right_eye, (IMG_SIZE - (38 + EYE_W // 2 - 1), 38 - EYE_W // 2 - 1, 54 - EYE_H // 2 - 1, IMG_SIZE - (54 + EYE_H // 2 - 1)), 'constant', 0)
        # out = torch.stack([f_left_eye_out,f_right_eye_out],dim=0)
        out = f_left_eye_out + f_right_eye_out
        # out = torch.max(out,dim=0)
        # return torch.max(torch.stack([f_left_eye_out,f_right_eye_out],dim=0),dim=0)[0]
        return out


class GlobalPathWay(nn.Module):
    def __init__(self):
        super(GlobalPathWay, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual6_1 = self.make_layer(_Residual_Block, 6, dim=128, kernel_size=7, padding=3, stride=1)
        self.residual6_2 = self.make_layer(_Residual_Block, 6, dim=128, kernel_size=7, padding=3, stride=1)
        self.residual = self.make_layer(_Residual_Block, 1, dim=192, kernel_size=7, padding=3, stride=1)
        self.bn = nn.BatchNorm2d(128, affine=True)
        self.bn_out = nn.BatchNorm2d(192, affine=True)
        self.in_out = nn.InstanceNorm2d(192, affine=True)
        self.instance = nn.InstanceNorm2d(128, affine=True)
        self.conv_out = nn.Conv2d(in_channels=192, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.out = nn.Tanh()
    
    def make_layer(self, block, num_of_layer, dim,kernel_size,stride,padding):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(dim=dim,kernel_size=kernel_size,stride=stride,padding=padding))
        return nn.Sequential(*layers)
    
    def forward(self, I112, local_feature):
        out = self.conv_in(I112)
        out = self.residual6_1(out)
        # out = self.residual6_2(out)
        out = self.instance(out)
        out = torch.cat([out, local_feature], dim=1)
        # out = self.residual(out)
        # out = self.conv_out(out)
        # out = self.in_out(out)
        out = self.conv_out(out)
        out = self.out(out)
        return out
    
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=8, img_size=112, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [_Residual_Block(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), _Residual_Block(dim=ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         nn.InstanceNorm2d(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.InstanceNorm2d(ngf)]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)
        self.conv_out = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.out = nn.Tanh()
        

    def forward(self, input,local_feature):
        x = self.DownBlock(input)

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
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)
        
        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)


        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x)
        out = self.UpBlock2(x)
        out = torch.cat([out, local_feature], dim=1)
        out = self.conv_out(out)
        out = self.out(out)

        return out, cam_logit, heatmap


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.local_pathway_left_eye = LocalPathWay()
        self.local_pathway_right_eye = LocalPathWay()
        self.global_pathway = ResnetGenerator()
        self.local_fuser = LocalFuser()
    
    def forward(self, I112, left_eye, right_eye):
        # path through local pathway
        left_fake_feature, left_fake = self.local_pathway_left_eye(left_eye)
        right_fake_feature, right_fake = self.local_pathway_right_eye(right_eye)
        
        # fusion
        local_feature = self.local_fuser(left_fake_feature, right_fake_feature)
        local_vision = self.local_fuser(left_fake, right_fake)
        local_input = self.local_fuser(left_eye, right_eye)
        
        # path through global pathway
        I112_fake, cam_logit, heatmap = self.global_pathway(I112, local_feature)
        return I112_fake, local_vision, local_input, left_fake, right_fake, cam_logit, heatmap
    
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=4):
        super(Discriminator, self).__init__()
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
    model = Discriminator().cuda()
    # model = Generator().cuda()
    input = Variable(torch.rand(3,3,112,112)).cuda()
    local_input = Variable(torch.rand(3,3,22,22)).cuda()
    local_feature = Variable(torch.rand(3,128,112,112)).cuda()
    # writter = SummaryWriter('./log')
    # I112_fake, local_vision, local_input, left_fake, right_fake, cam_logit, heatmap = model(input,local_input,local_input)
    out,cam_logit, heatmap = model(local_input)
    print(model(input)[0].size())