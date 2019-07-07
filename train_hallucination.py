import time
import numpy as np
import torch
import random
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torch.autograd import Variable

from utils import *
import config
from DataLoader import Dataset

from model.model import Discriminator, Generator

def weights_init(m):
    for each in m.modules():
        if isinstance(each,nn.Conv2d):
            nn.init.xavier_uniform_(each.weight.data)
            if each.bias is not None:
                each.bias.data.zero_()
        elif isinstance(each,nn.BatchNorm2d):
            each.weight.data.fill_(1)
            each.bias.data.zero_()
        elif isinstance(each,nn.Linear):
            nn.init.xavier_uniform_(each.weight.data)
            each.bias.data.zero_()
    

if __name__=='__main__':

    if config.train['random_seed'] is None:
        config.train['random_seed'] = random.randint(1,10000)
    print('Random seed: ',config.train['random_seed'])
    random.seed(config.train['random_seed'])
    torch.manual_seed(config.train['random_seed'])
    
    if config.train['cuda']:
        torch.cuda.manual_seed_all(config.train['random_seed'])
        
    # Dataloader
    trainset = Dataset()
    trainLoader = data.DataLoader(trainset,batch_size=config.train['batch_size'],shuffle=True,num_workers=config.train['num_workers'])
    
    # Model
    D_N = Discriminator()
    D_V = Discriminator()
    G_N2V = Generator()
    G_V2N = Generator()
    
    D_N.apply(weights_init)
    D_V.apply(weights_init)
    G_N2V.apply(weights_init)
    G_V2N.apply(weights_init)
    
    # Optimizer
    optimizer_D_N = torch.optim.Adam(D_N.parameters(),lr=config.train['lr_D_N'],betas=(config.train['beta1_D_N'],config.train['beta2_D_N']),weight_decay=config.train['weight_decay_D_N'])
    optimizer_D_V = torch.optim.Adam(D_V.parameters(),lr=config.train['lr_D_V'],betas=(config.train['beta1_D_V'],config.train['beta2_D_V']),weight_decay=config.train['weight_decay_D_V'])
    optimizer_G_V2N = torch.optim.Adam(G_V2N.parameters(),lr=config.train['lr_G_V2N'],betas=(config.train['beta1_G_V2N'],config.train['beta2_G_N2V']),weight_decay=config.train['weight_decay_G_V2N'])
    optimizer_G_N2V = torch.optim.Adam(G_N2V.parameters(),lr=config.train['lr_G_N2V'],betas=(config.train['beta1_G_N2V'],config.train['beta2_G_N2V']),weight_decay=config.train['weight_decay_G_N2V'])
    
    if config.train['resume_G_N2V']:
        checkpoint = torch.load(config.train['resume_G_N2V'])
        G_N2V.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        
    if config.train['resume_G_V2N']:
        checkpoint = torch.load(config.train['resume_G_V2N'])
        G_V2N.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        
        
        
        
    
