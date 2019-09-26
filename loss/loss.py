import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class MSELossFunc(nn.Module):
    def __init__(self):
        super(MSELossFunc,self).__init__()
        # self.input = input
        # self.target = target
        
    def forward(self,input,target):
        loss = torch.mean(torch.pow((input.float()-target.float()),2))*97.0
        return loss

class MSELoss_Landmark(nn.Module):
    def __init__(self):
        super(MSELoss_Landmark,self).__init__()
    def forward(self,input,target):
        # weight = np.ones((input.size()[0],input.size()[1],input.size()[2],input.size()[3]))
        # weight[:,0:20,:,:] = weight[:,0:20,:,:]*0.7
        # weight = np.ones((input.size()[0], input.size()[1], input.size()[2], input.size()[3]))
        # weight[:, 0:20, :, :] = weight[:, 0:20, :, :] * 1.2
        # weight = torch.from_numpy(weight.astype(np.float32)).cuda()
        # loss = torch.sum((torch.mean(((input.float()-target.float()).mul(weight))**2)))
        # input = input.mul(weight)
        input = torch.sum(input,dim=1)
        # pdb.set_trace()
        # loss = torch.sum(torch.pow(((input.float() - target.float()).mul(weight)) ,2))/97.0
        loss = torch.mean(torch.pow(((input.float() - target.float())), 2)) * 97.0
        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self,weight=None):
        super(CrossEntropyLoss2d,self).__init__()
        self.loss = nn.NLLLoss()
    '''
    def forward(self ,output,target,weight=None,size_average=True):
        n,c,h,w = output.size()
        nt,ht,wt = target.squeeze().size()
        m = nn.LogSoftmax(dim=1)
        # example
        N,C = 5,4
        data = torch.randn(N,16,10,10)
        conv = nn.Conv2d(16,C,(3,3))
        output1 = conv(data)
        out1 = m(output1)
        target1 = torch.empty(N,8,8,dtype=torch.long).random_(0,C)
        # network
        out = m(output)
        pdb.set_trace()
        output = output.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
        target = target.view(-1)
        
        loss = F.cross_entropy(output,target,weight=weight,size_average = size_average,\
               ignore_index=250)
        return loss
    '''

    def forward(self,outputs,targets):
        return self.loss(F.log_softmax(outputs,1),torch.squeeze(targets))
    
class MMD(nn.Module):
    def __init__(self):
        super(MMD,self).__init__()
        
    def gaussian_kernel(self,source,target,kernel_mul=2.0,kernel_num=5,fix_sigma=1.0):
        '''
        Transfer source and target data to kernel matrix
        :param source: Source
        :param target: Target
        :param kernel_mul:
        :param kernel_num: number of kernel
        :param fix_sigma: sigma value of different gaussian kernel
        :return:
        '''
        source = source.view(source.size()[0],source.size()[1])/(source.max()-source.min())
        target = target.view(source.size()[0],source.size()[1])/(source.max()-source.min())
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source,target],dim=0)
        # total0 = total.unsqueeze(0).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)),int(total.size(2)),int(total.size(3)))
        # total1 = total.unsqueeze(1).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)),int(total.size(2)),int(total.size(3)))
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).mean(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            
        # 以fix_sigma为中值，kernel_mul为倍数，取kernel_num个bandwidth值
        # 比如fix_sigma=1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num//2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        # gaussian kernel function
        # kernel_val = [torch.exp(-L2_distance/bandwidth_temp) for bandwidth_temp in bandwidth_list]
        kernel_val = [-L2_distance / bandwidth_temp for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
        
    def forward(self,source,target,kernel_mul=2.0,kernel_num=5,fix_sigma=None):
        """
        calculate MMD distance between source and target
        :param source:
        :param target:
        :return: MMD Loss
        """
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source,target,kernel_mul,kernel_num,fix_sigma)
        XX = kernels[:batch_size,:batch_size]
        YY = kernels[batch_size:,batch_size:]
        XY = kernels[:batch_size,batch_size:]
        YX = kernels[batch_size:,:batch_size]
        loss = torch.mean(XX+YY-XY-YX)
        return loss
        
    
