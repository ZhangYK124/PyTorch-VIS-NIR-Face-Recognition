import torch
import numpy as np

import tensorflow as tf
import tensorboardX
from tensorboardX import SummaryWriter

import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

import pdb


class _Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Residual_Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.conv1 = OctConv(ch_in=in_channels,ch_out=out_channels,kernel_size=3,stride=1,alphas=(0,0.5))
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.batch_channel = int(160*out_channels/64)
        # self.in1 = nn.InstanceNorm2d(self.batch_channel, affine=True)
        self.relu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.conv2 = OctConv(ch_in=in_channels, ch_out=out_channels, kernel_size=3, stride=1, alphas=(0.5, 0))
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu_out = nn.PReLU(out_channels)
    
    def forward(self, x):
        identity_data = x
        out = self.conv1(x)
        out = self.relu(self.in1(out))
        out = self.conv2(out)
        out = self.in2(out)
        out = torch.add(out, identity_data)
        out = self.relu_out(out)
        return out
    
class LocalPathWay(nn.Module):
    def __init__(self):
        super(LocalPathWay,self).__init__()
        self.conv = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.residual = self.make_layer(_Residual_Block, 3, in_channel=64, out_channel=64)
        self.bn = nn.BatchNorm2d(64,affine=True)
        self.instance = nn.InstanceNorm2d(64,affine=True)
        self.conv_out = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1,bias=False)

    def make_layer(self, block, num_of_layer,in_channel, out_channel):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channel,out_channel))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.conv(x)
        out = self.residual(out)
        out = self.residual(out)
        out_feature = self.instance(out)
        local_img = self.conv_out(out_feature)
        return out_feature, local_img
    
class LocalFuser(nn.Module):
    '''
    Landmark coordinate:
    x       y
    
    '''
    def __init__(self):
        super(LocalFuser,self).__init__()
    def forward(self,f_left_eye,f_right_eye):
        EYE_W, EYE_H = 22,22
        IMG_SIZE = 112
        f_left_eye_out = torch.nn.functional.pad(f_left_eye,(38 - EYE_W//2  - 1 ,IMG_SIZE - (38 + EYE_W//2 - 1) ,54 - EYE_H//2 - 1, IMG_SIZE - (54 + EYE_H//2 - 1)),'constant',0)
        f_right_eye_out = torch.nn.functional.pad(f_right_eye,(IMG_SIZE - (38 + EYE_W//2 - 1) ,38 - EYE_W//2  - 1  ,54 - EYE_H//2 - 1, IMG_SIZE - (54 + EYE_H//2 - 1)),'constant',0)
        # out = torch.stack([f_left_eye_out,f_right_eye_out],dim=0)
        out = f_left_eye_out+f_right_eye_out
        # out = torch.max(out,dim=0)
        # return torch.max(torch.stack([f_left_eye_out,f_right_eye_out],dim=0),dim=0)[0]
        return out
    
class GlobalPathWay(nn.Module):
    def __init__(self):
        super(GlobalPathWay,self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.residual6_1 = self.make_layer(_Residual_Block, 6, in_channel=64, out_channel=64)
        self.residual6_2 = self.make_layer(_Residual_Block,6,in_channel=64,out_channel=64)
        self.residual3 = self.make_layer(_Residual_Block, 3, in_channel=128, out_channel=128)
        self.bn = nn.BatchNorm2d(128,affine=True)
        self.instance = nn.InstanceNorm2d(128,affine=True)
        self.conv_out = nn.Conv2d(in_channels=128,out_channels=3,kernel_size=3,stride=1,padding=1,bias=False)

        
    def make_layer(self, block, num_of_layer,in_channel, out_channel):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channel,out_channel))
        return nn.Sequential(*layers)
        
    def forward(self, I112,local_feature):
        out = self.conv_in(I112)
        out = self.residual6_1(out)
        # out = self.residual6_2(out)
        out = torch.cat([out,local_feature],dim=1)
        out = self.residual3(out)
        out = self.instance(out)
        out = self.conv_out(out)
        return out
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.local_pathway_left_eye = LocalPathWay()
        self.local_pathway_right_eye = LocalPathWay()
        self.global_pathway = GlobalPathWay()
        self.local_fuser = LocalFuser()
    
    def forward(self,I112,left_eye,right_eye):
        # path through local pathway
        left_fake_feature,left_fake = self.local_pathway_left_eye(left_eye)
        right_fake_feature, right_fake = self.local_pathway_right_eye(right_eye)
        
        # fusion
        local_feature = self.local_fuser(left_fake_feature,right_fake_feature)
        local_vision = self.local_fuser(left_fake,right_fake)
        local_input = self.local_fuser(left_eye,right_eye)
        
        # path through global pathway
        I112_fake = self.global_pathway(I112,local_feature)
        return I112_fake, local_vision,local_input,left_fake,right_fake
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.residual3 = self.make_layer(_Residual_Block, 3, in_channel=64, out_channel=64)
        self.bn = nn.BatchNorm2d(64,affine=True)
        self.instance = nn.InstanceNorm2d(64,affine=True)
        self.out_layer = nn.Sigmoid()
        self.conv_out = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
        
    def make_layer(self, block, num_of_layer,in_channel, out_channel):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channel,out_channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.residual3(out)
        out = self.instance(out)
        out = self.conv_out(out)
        return F.avg_pool2d(out,out.size()[2:]).view(out.size()[0],-1)
    
if __name__ == '__main__':
    model = Discriminator().cuda()
    # model = Generator().cuda()
    input = Variable(torch.rand(3,3,112,112)).cuda()
    writter = SummaryWriter('./log')
    output = model(input)
    print(model(input)[0].size())
        
        
