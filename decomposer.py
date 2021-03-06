import time
import os
import numpy as np
import torch
import random
from random import choice
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torch.autograd import Variable

import itertools
from tqdm import tqdm as tqdm
import datetime
from tensorboardX import SummaryWriter

from utils import *
import config_decomposer as config
from DataLoader_Star import Dataset

from model.model_decomposer import Discriminator_CAM as Discriminator
from model.model_decomposer import Discriminator as Discriminator_Domain
from model.model_decomposer import Integrator, Intrinsic_Encoder, Style_Encoder
from loss.loss import MMD


def weights_init(m):
    for each in m.modules():
        if isinstance(each, nn.Conv2d):
            nn.init.xavier_uniform_(each.weight.data)
            if each.bias is not None:
                each.bias.data.zero_()
        elif isinstance(each, nn.BatchNorm2d):
            each.weight.data.fill_(1)
        #     each.bias.data.zero_()
        # elif isinstance(each, nn.Linear):
        #     nn.init.xavier_uniform_(each.weight.data)
        #     each.bias.data.zero_()


def label2onehot(labels, c):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size)
    out[np.arange(batch_size)] = c
    return out.long().cuda()


if __name__ == '__main__':
    
    if config.train['random_seed'] is None:
        config.train['random_seed'] = random.randint(1, 10000)
    print('Random seed: ', config.train['random_seed'])
    random.seed(config.train['random_seed'])
    torch.manual_seed(config.train['random_seed'])
    
    if not os.path.exists(config.train['logs']):
        os.mkdir(config.train['logs'])
    if not os.path.exists(config.train['checkpoint']):
        os.mkdir(config.train['checkpoint'])
    if not os.path.exists(config.train['out']):
        os.mkdir(config.train['out'])
    
    if config.train['cuda']:
        torch.cuda.manual_seed_all(config.train['random_seed'])
    
    # Dataloader
    trainset = Dataset()
    trainLoader = data.DataLoader(trainset, batch_size=config.train['batch_size'], shuffle=True,
                                  num_workers=config.train['num_workers'])
    
    # Model
    D_DOMAIN = Discriminator_Domain()
    D_VIS = Discriminator()
    D_NIR = Discriminator()
    D_SKETCH = Discriminator()
    Integrator = Integrator()
    Style_Encoder = Style_Encoder()
    Intrinsic_Encoder = Intrinsic_Encoder()
    
    # Losses
    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    cross_entropy = torch.nn.CrossEntropyLoss()
    mmd = MMD()
    
    # Optimizer
    optimizer_D_DOMAIN = torch.optim.Adam(D_DOMAIN.parameters(), lr=config.train['lr_D'],
                                          betas=(config.train['beta1_D'], config.train['beta2_D']),
                                          weight_decay=config.train['weight_decay_D'])
    optimizer_D_VIS = torch.optim.Adam(D_VIS.parameters(), lr=config.train['lr_D'],
                                       betas=(config.train['beta1_D'], config.train['beta2_D']),
                                       weight_decay=config.train['weight_decay_D'])
    optimizer_D_NIR = torch.optim.Adam(D_NIR.parameters(), lr=config.train['lr_D'],
                                       betas=(config.train['beta1_D'], config.train['beta2_D']),
                                       weight_decay=config.train['weight_decay_D'])
    optimizer_D_SKETCH = torch.optim.Adam(D_SKETCH.parameters(), lr=config.train['lr_D'],
                                          betas=(config.train['beta1_D'], config.train['beta2_D']),
                                          weight_decay=config.train['weight_decay_D'])
    # optimizer_G = torch.optim.Adam(itertools.chain(Style_Encoder.parameters(), Intrinsic_Encoder.parameters(), Integrator.parameters()),
    #                                lr=config.train['lr_G'],
    #                                betas=(config.train['beta1_G'], config.train['beta2_G']),
    #                                weight_decay=config.train['weight_decay_G'])
    optimizer_Style = torch.optim.Adam(Style_Encoder.parameters(),
                                       lr = config.train['lr_G'],
                                       betas=(config.train['beta1_G'],config.train['beta2_G']),
                                       weight_decay=config.train['weight_decay_G'])
    optimizer_Intrinsic = torch.optim.Adam(Intrinsic_Encoder.parameters(),
                                       lr=config.train['lr_G'],
                                       betas=(config.train['beta1_G'], config.train['beta2_G']),
                                       weight_decay=config.train['weight_decay_G'])
    optimizer_Integrator = torch.optim.Adam(Integrator.parameters(),
                                       lr=config.train['lr_G'],
                                       betas=(config.train['beta1_G'], config.train['beta2_G']),
                                       weight_decay=config.train['weight_decay_G'])
    
    if config.train['resume_Style_Encoder']:
        checkpoint = torch.load(config.train['resume_Style_Encoder'])
        Style_Encoder.load_state_dict(checkpoint['state_dict_Style_Encoder'])
        start_epoch = checkpoint['epoch_Style_Encoder'] + 1
        optim_checkpoint = torch.load(config.train['resume_optim_Style'])
        optimizer_Style.load_state_dict(optim_checkpoint)
        for state in optimizer_Style.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        checkpoint = torch.load(config.train['resume_Intrinsic_Encoder'])
        Intrinsic_Encoder.load_state_dict(checkpoint['state_dict_Intrinsic_Encoder'])
        start_epoch = checkpoint['epoch_Intrinsic_Encoder'] + 1
        optim_checkpoint = torch.load(config.train['resume_optim_Intrinsic'])
        optimizer_Intrinsic.load_state_dict(optim_checkpoint)
        for state in optimizer_Intrinsic.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        checkpoint = torch.load(config.train['resume_Integrator'])
        Integrator.load_state_dict(checkpoint['state_dict_Integrator'])
        start_epoch = checkpoint['epoch_Integrator'] + 1
        optim_checkpoint = torch.load(config.train['resume_optim_Integrator'])
        optimizer_Integrator.load_state_dict(optim_checkpoint)
        for state in optimizer_Integrator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    
    else:
        Integrator.apply(weights_init)
        Intrinsic_Encoder.apply(weights_init)
        Style_Encoder.apply(weights_init)
        start_epoch = 0
    
    if config.train['resume_D_VIS']:
        checkpoint = torch.load(config.train['resume_D_VIS'])
        D_VIS.load_state_dict(checkpoint['state_dict_D_VIS'])
        
        checkpoint = torch.load(config.train['resume_D_NIR'])
        D_NIR.load_state_dict(checkpoint['state_dict_D_NIR'])
        
        checkpoint = torch.load(config.train['resume_D_SKETCH'])
        D_SKETCH.load_state_dict(checkpoint['state_dict_D_SKETCH'])
        
        checkpoint = torch.load(config.train['resume_D_DOMAIN'])
        D_DOMAIN.load_state_dict(checkpoint['state_dict_D_DOMAIN'])
        
        # start_epoch = checkpoint['epoch_D_DOMAIN'] + 1
        
        optim_checkpoint_VIS = torch.load(config.train['resume_optim_D_VIS'])
        optimizer_D_VIS.load_state_dict(optim_checkpoint_VIS)
        
        for state in optimizer_D_VIS.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        optim_checkpoint_NIR = torch.load(config.train['resume_optim_D_NIR'])
        optimizer_D_NIR.load_state_dict(optim_checkpoint_NIR)
        
        for state in optimizer_D_NIR.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        optim_checkpoint_SKETCH = torch.load(config.train['resume_optim_D_SKETCH'])
        optimizer_D_SKETCH.load_state_dict(optim_checkpoint_SKETCH)
        
        for state in optimizer_D_SKETCH.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        optim_checkpoint_DOMAIN = torch.load(config.train['resume_optim_D_DOMAIN'])
        optimizer_D_DOMAIN.load_state_dict(optim_checkpoint_DOMAIN)
        
        for state in optimizer_D_DOMAIN.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    
    else:
        D_VIS.apply(weights_init)
        D_NIR.apply(weights_init)
        D_SKETCH.apply(weights_init)
        start_epoch = 0
    
    if config.train['cuda']:
        D_VIS.cuda()
        D_NIR.cuda()
        D_SKETCH.cuda()
        D_DOMAIN.cuda()
        Intrinsic_Encoder.cuda()
        Style_Encoder.cuda()
        Integrator.cuda()
        l1_loss.cuda()
        mse_loss.cuda()
        cross_entropy.cuda()
    
    # ################################       LR Schedulers           ################################
    # lr_schedule_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(config.train['epochs'],
    #                                                                                   config.train['lr_G_decay_rate'],
    #                                                                                   config.train['lr_G_decay_epoch'],
    #                                                                                   config.train['lr_G']).step)

    lr_schedule_Style = torch.optim.lr_scheduler.LambdaLR(optimizer_Style, lr_lambda=LambdaLR(config.train['epochs'],
                                                                                      config.train['lr_G_decay_rate'],
                                                                                      config.train['lr_G_decay_epoch'],
                                                                                      config.train['lr_G']).step)

    lr_schedule_Intrinsic = torch.optim.lr_scheduler.LambdaLR(optimizer_Intrinsic, lr_lambda=LambdaLR(config.train['epochs'],
                                                                                      config.train['lr_G_decay_rate'],
                                                                                      config.train['lr_G_decay_epoch'],
                                                                                      config.train['lr_G']).step)

    lr_schedule_Integrator = torch.optim.lr_scheduler.LambdaLR(optimizer_Integrator, lr_lambda=LambdaLR(config.train['epochs'],
                                                                                      config.train['lr_G_decay_rate'],
                                                                                      config.train['lr_G_decay_epoch'],
                                                                                      config.train['lr_G']).step)
    
    lr_schedule_D_VIS = torch.optim.lr_scheduler.LambdaLR(optimizer_D_VIS, lr_lambda=LambdaLR(config.train['epochs'],
                                                                                              config.train['lr_D_decay_rate'],
                                                                                              config.train['lr_D_decay_epoch'],
                                                                                              config.train['lr_D']).step)
    
    lr_schedule_D_NIR = torch.optim.lr_scheduler.LambdaLR(optimizer_D_NIR, lr_lambda=LambdaLR(config.train['epochs'],
                                                                                              config.train['lr_D_decay_rate'],
                                                                                              config.train['lr_D_decay_epoch'],
                                                                                              config.train['lr_D']).step)
    
    lr_schedule_D_SKETCH = torch.optim.lr_scheduler.LambdaLR(optimizer_D_SKETCH, lr_lambda=LambdaLR(config.train['epochs'],
                                                                                                    config.train['lr_D_decay_rate'],
                                                                                                    config.train['lr_D_decay_epoch'],
                                                                                                    config.train['lr_D']).step)
    
    lr_schedule_D_DOMAIN = torch.optim.lr_scheduler.LambdaLR(optimizer_D_DOMAIN, lr_lambda=LambdaLR(config.train['epochs'],
                                                                                                    config.train['lr_D_decay_rate'],
                                                                                                    config.train['lr_D_decay_epoch'],
                                                                                                    config.train['lr_D']).step)
    
    prev_time = time.time()
    
    count = int(len(trainLoader) // config.train['save_img_num'])
    
    vis_buffer = ReplayBuffer()
    nir_buffer = ReplayBuffer()
    sketch_buffer = ReplayBuffer()
    
    print('***************** Save Image Iteration: {} ********************'.format(count))
    # writer_loss_G_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_G')
    writer_loss_Intrinsic_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_Intrinsic')
    writer_loss_Style_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_Style')
    writer_loss_Integrator_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_Integrator')
    writer_loss_D_VIS_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D_VIS')
    writer_loss_D_NIR_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D_NIR')
    writer_loss_D_SKETCH_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D_SKETCH')
    writer_loss_D_DOMAIN_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D_DOMAIN')
    
    # Training
    if config.train['if_train']:
        for epoch in range(start_epoch, config.train['epochs']):
            # writer_loss_G_steps = SummaryWriter(config.train['logs'] + 'steps/loss_G')
            writer_loss_Intrinsic_steps = SummaryWriter(config.train['logs'] + 'steps/loss_Intrinsic')
            writer_loss_Style_steps = SummaryWriter(config.train['logs'] + 'steps/loss_Style')
            writer_loss_Integrator_steps = SummaryWriter(config.train['logs'] + 'steps/loss_Integrator')
            writer_loss_D_VIS_steps = SummaryWriter(config.train['logs'] + 'steps/loss_D_VIS')
            writer_loss_D_NIR_steps = SummaryWriter(config.train['logs'] + 'steps/loss_D_NIR')
            writer_loss_D_SKETCH_steps = SummaryWriter(config.train['logs'] + 'steps/loss_D_SKETCH')
            writer_loss_D_DOMAIN_steps = SummaryWriter(config.train['logs'] + 'steps/loss_D_DOMAIN')
            
            D_VIS.train()
            D_NIR.train()
            D_SKETCH.train()
            D_DOMAIN.train()
            Integrator.train()
            Style_Encoder.train()
            Intrinsic_Encoder.train()
            
            batch_time = AverageMeter()
            data_time = AverageMeter()
            # losses_G = AverageMeter()
            losses_Intrinsic = AverageMeter()
            losses_Style = AverageMeter()
            losses_Integrator = AverageMeter()
            losses_D_VIS = AverageMeter()
            losses_D_NIR = AverageMeter()
            losses_D_SKETCH = AverageMeter()
            losses_D_DOMAIN = AverageMeter()
            
            lr_G = LambdaLR(config.train['epochs'], config.train['lr_G_decay_rate'],
                            config.train['lr_G_decay_epoch'], config.train['lr_G']).step(epoch)
            lr_D = LambdaLR(config.train['epochs'], config.train['lr_D_decay_rate'],
                            config.train['lr_D_decay_epoch'], config.train['lr_D']).step(epoch)
            
            end_time = time.time()
            
            bar = Bar('Processing: ', max=len(trainLoader))
            
            for i, batch in enumerate(trainLoader):
                for k in batch:
                    batch[k] = batch[k].cuda()
                
                step = 0
                src_list = ['sketch', 'nir', 'vis']
                # for src in ['sketch', 'nir', 'vis']:
                if True:
                    src = choice(src_list)
                    if src == 'vis':
                        c_src = 0
                        D_X = D_VIS
                        optimizer_D_X = optimizer_D_VIS
                        src_buffer = vis_buffer
                        extra = 1.0
                        losses_D_X = losses_D_VIS
                        writer_loss_D_X_steps = writer_loss_D_VIS_steps
                    
                    elif src == 'nir':
                        c_src = 1
                        D_X = D_NIR
                        optimizer_D_X = optimizer_D_NIR
                        src_buffer = nir_buffer
                        extra = 1.0
                        losses_D_X = losses_D_NIR
                        writer_loss_D_X_steps = writer_loss_D_NIR_steps
                    
                    elif src == 'sketch':
                        c_src = 2
                        D_X = D_SKETCH
                        optimizer_D_X = optimizer_D_SKETCH
                        src_buffer = sketch_buffer
                        extra = 3.0
                        losses_D_X = losses_D_SKETCH
                        writer_loss_D_X_steps = writer_loss_D_SKETCH_steps
                    
                    x_real = batch[src]
                    target_list = ['vis', 'sketch', 'nir']
                    while src in target_list:
                        target_list.remove(src)
                    
                    # for tgt in target_list:
                    if True:
                        tgt = choice(target_list)
                        if tgt == 'vis':
                            c_tgt = 0
                            D_Y = D_VIS
                            optimizer_D_Y = optimizer_D_VIS
                            tgt_buffer = vis_buffer
                            losses_D_Y = losses_D_VIS
                            writer_loss_D_Y_steps = writer_loss_D_VIS_steps
                        
                        elif tgt == 'nir':
                            c_tgt = 1
                            D_Y = D_NIR
                            optimizer_D_Y = optimizer_D_NIR
                            tgt_buffer = nir_buffer
                            losses_D_Y = losses_D_NIR
                            writer_loss_D_Y_steps = writer_loss_D_NIR_steps
                        
                        elif tgt == 'sketch':
                            c_tgt = 2
                            D_Y = D_SKETCH
                            optimizer_D_Y = optimizer_D_SKETCH
                            tft_buffer = sketch_buffer
                            extra = 1.0
                            losses_D_Y = losses_D_SKETCH
                            writer_loss_D_Y_steps = writer_loss_D_SKETCH_steps
                        
                        y_real = batch[tgt]
                        
                        data_time.update(time.time() - end_time)
                        
                        # =================================================================================== #
                        #                               3. Train the generator                                #
                        # =================================================================================== #
                        if (i + 0) % config.train['generator_steps'] == 0:
                            x_intrinsic = Intrinsic_Encoder(x_real)
                            y_intrinsic = Intrinsic_Encoder(y_real)
                            
                            x_style, x_style_logit = Style_Encoder(x_real)
                            y_style, y_style_logit = Style_Encoder(y_real)
                            
                            x_rec, x_rec_heatmap, x_cam_logit = Integrator(x_intrinsic, x_style)
                            y_rec, y_rec_heatmap, y_cam_logit = Integrator(y_intrinsic, y_style)
                            
                            xy, xy_heatmap, xy_cam_logit = Integrator(x_intrinsic, y_style)
                            yx, yx_heatmap, yx_cam_logit = Integrator(y_intrinsic, x_style)
                            
                            xy_intrinsic = Intrinsic_Encoder(xy)
                            yx_intrinsic = Intrinsic_Encoder(yx)
                            # yx_intrinsic = Intrinsic_Encoder(x_real)
                            
                            xy_style, xy_style_logit = Style_Encoder(xy)
                            yx_style, yx_style_logit = Style_Encoder(yx)
                            # yx_style, yx_style_logit = Style_Encoder(x_real)
                            
                            x_recon, x_recon_heatmap, x_recon_cam_logit = Integrator(xy_intrinsic, yx_style)
                            y_recon, y_recon_heatmap, y_recon_cam_logit = Integrator(yx_intrinsic, xy_style)

                            # x_recon, x_recon_heatmap, x_recon_cam_logit = Integrator(xy_intrinsic, x_style)
                            # y_recon, y_recon_heatmap, y_recon_cam_logit = Integrator(yx_intrinsic, y_style)
                            
                            # intrinsic loss
                            g_intrinsic = mse_loss(x_intrinsic, xy_intrinsic) + mse_loss(y_intrinsic, yx_intrinsic)
                            
                            # reconstruction loss
                            g_recon =l1_loss(x_real, x_rec) +l1_loss(y_real, y_rec)
                            
                            # cycle consistency loss
                            g_cycle =l1_loss(x_real, x_recon) +l1_loss(y_real, y_recon)
                            
                            # mmd loss
                            g_mmd = mmd(x_intrinsic, y_intrinsic)
                                    # + mmd(xy_intrinsic, yx_intrinsic) + mmd(x_intrinsic, yx_intrinsic) + mmd(y_intrinsic, xy_intrinsic)
                            
                            # domain_cls1 = D_DOMAIN(x_intrinsic, y_intrinsic)
                            # domain_cls2 = D_DOMAIN(xy_intrinsic, yx_intrinsic)
                            # domain_cls3 = D_DOMAIN(x_intrinsic, yx_intrinsic)
                            # domain_cls4 = D_DOMAIN(y_intrinsic, xy_intrinsic)
                            # g_domain = mse_loss(domain_cls1, torch.ones_like(domain_cls1).cuda()) \
                            #            + mse_loss(domain_cls2, torch.ones_like(domain_cls2).cuda()) \
                            #            + mse_loss(domain_cls2, torch.ones_like(domain_cls2).cuda()) \
                            #            + mse_loss(domain_cls2, torch.ones_like(domain_cls2).cuda())
                            
                            # adv loss & cam loss
                            # x_rec_adv, x_rec_adv_cam_logit, _ = D_X(x_rec)
                            # y_rec_adv, y_rec_adv_cam_logit, _ = D_Y(y_rec)
                            
                            xy_adv, xy_adv_cam_logit, _ = D_Y(xy)
                            yx_adv, yx_adv_cam_logit, _ = D_X(yx)
                            
                            g_adv = mse_loss(xy_adv, torch.ones_like(xy_adv).cuda()) + mse_loss(yx_adv, torch.ones_like(yx_adv).cuda())
                            g_cam = mse_loss(xy_adv_cam_logit, torch.ones_like(xy_adv_cam_logit).cuda()) \
                                    + mse_loss(yx_adv_cam_logit, torch.ones_like(yx_adv_cam_logit).cuda())
                            
                            # style cls loss
                            x_cls_label = label2onehot(x_style_logit, c_src)
                            y_cls_label = label2onehot(y_style_logit, c_tgt)
                            
                            g_cls = cross_entropy(x_style_logit, x_cls_label) + cross_entropy(y_style_logit, y_cls_label) \
                                    + cross_entropy(xy_style_logit, y_cls_label) * 0.0 + cross_entropy(yx_style_logit, x_cls_label) * 0.0
                            
                            # total loss
                            # g_loss = g_recon * 10.0 * 255.0 + g_cycle * 10.0 * 255.0 + g_adv * 1.0 * 0.0 + g_cam * 10.0 * 0.0 + g_cls * 1.0 + g_domain * 0.0 + g_intrinsic * 1.0 * 0.0

                            # optimizer_G.zero_grad()
                            # g_loss.backward()
                            # optimizer_G.step()
                            
                            intrinsic_loss = g_recon * 10.0 + g_cycle * 10.0 + g_adv * 0.1 + g_cam * 1.0 + g_mmd * 0.0 + g_intrinsic * 0.1
                            style_loss = g_cls
                            integrator_loss = g_recon * 10.0 + g_cycle * 10.0 + g_adv * 0.1 + g_cam * 1.0
                            
                            optimizer_Intrinsic.zero_grad()
                            intrinsic_loss.backward(retain_graph=True)
                            optimizer_Intrinsic.step()
                            
                            optimizer_Style.zero_grad()
                            style_loss.backward(retain_graph=True)
                            optimizer_Style.step()
                            
                            optimizer_Integrator.zero_grad()
                            integrator_loss.backward()
                            optimizer_Integrator.step()
                            
                        
                        # =================================================================================== #
                        #                             2. Train the discriminator                              #
                        # =================================================================================== #
                        # x_rec_adv, x_rec_adv_cam_logit, _ = D_X(x_rec.detach())
                        # y_rec_adv, y_rec_adv_cam_logit, _ = D_Y(y_rec.detach())
                        
                        xy_adv, xy_adv_cam_logit, _ = D_Y(xy.detach())
                        yx_adv, yx_adv_cam_logit, _ = D_X(yx.detach())
                        
                        d_X_adv_fake = mse_loss(yx_adv, torch.zeros_like(yx_adv).cuda())
                        
                        d_X_cam_fake = mse_loss(yx_adv_cam_logit, torch.zeros_like(yx_adv_cam_logit))
                        
                        d_Y_adv_fake = mse_loss(xy_adv, torch.zeros_like(xy_adv).cuda())
                        
                        d_Y_cam_fake = mse_loss(xy_adv_cam_logit, torch.zeros_like(xy_adv_cam_logit).cuda())
                        
                        d_loss_x_fake = d_X_adv_fake * 1.0 + d_X_cam_fake * 10.0
                        d_loss_y_fake = d_Y_adv_fake * 1.0 + d_Y_cam_fake * 10.0
                        
                        # Backward and optimize.
                        optimizer_D_X.zero_grad()
                        d_loss_x_fake.backward()
                        optimizer_D_X.step()
                        
                        optimizer_D_Y.zero_grad()
                        d_loss_y_fake.backward()
                        optimizer_D_Y.step()
                        
                        x_adv, x_adv_cam_logit, _ = D_X(x_real)
                        y_adv, y_adv_cam_logit, _ = D_Y(y_real)
                        
                        d_X_adv_real = mse_loss(x_adv, torch.ones_like(x_adv).cuda())
                        
                        d_X_cam_real = mse_loss(x_adv_cam_logit, torch.ones_like(x_adv_cam_logit))
                        
                        d_Y_adv_real = mse_loss(y_adv, torch.ones_like(y_adv).cuda())
                        
                        d_Y_cam_real = mse_loss(y_adv_cam_logit, torch.ones_like(y_adv_cam_logit))
                        
                        d_loss_x_real = d_X_adv_real * 1.0 + d_X_cam_real * 10.0
                        d_loss_y_real = d_Y_adv_real * 1.0 + d_Y_cam_real * 10.0
                        
                        # Backward and optimize.
                        optimizer_D_X.zero_grad()
                        d_loss_x_real.backward()
                        optimizer_D_X.step()
                        
                        optimizer_D_Y.zero_grad()
                        d_loss_y_real.backward()
                        optimizer_D_Y.step()
                        
                        d_loss_x = d_X_adv_real + d_X_adv_fake
                        d_loss_y = d_Y_adv_real + d_Y_adv_fake
                        
                        '''
                        # =================================================================================== #
                        #                          3. Train the domain discriminator                          #
                        # =================================================================================== #
                        x_intrinsic = Intrinsic_Encoder(x_real)
                        y_intrinsic = Intrinsic_Encoder(y_real)

                        x_style, x_style_logit = Style_Encoder(x_real)
                        y_style, y_style_logit = Style_Encoder(y_real)

                        # x_rec, x_rec_heatmap, x_cam_logit = Integrator(x_intrinsic, x_style)
                        # y_rec, y_rec_heatmap, y_cam_logit = Integrator(y_intrinsic, y_style)

                        xy, xy_heatmap, xy_cam_logit = Integrator(x_intrinsic, y_style)
                        yx, yx_heatmap, yx_cam_logit = Integrator(y_intrinsic, x_style)

                        xy_intrinsic = Intrinsic_Encoder(xy)
                        yx_intrinsic = Intrinsic_Encoder(yx)

                        if i % 10 == 0:
                            domain_cls1 = D_DOMAIN(x_intrinsic, y_intrinsic)
                            domain_cls2 = D_DOMAIN(xy_intrinsic, yx_intrinsic)
                            domain_cls3 = D_DOMAIN(x_intrinsic, yx_intrinsic)
                            domain_cls4 = D_DOMAIN(y_intrinsic, xy_intrinsic)
                            d_domain = mse_loss(domain_cls1, torch.zeros_like(domain_cls1).cuda()) \
                                       + mse_loss(domain_cls2, torch.zeros_like(domain_cls2).cuda()) \
                                       + mse_loss(domain_cls2, torch.zeros_like(domain_cls2).cuda()) \
                                       + mse_loss(domain_cls2, torch.zeros_like(domain_cls2).cuda())
                            d_domain1 = d_domain * 1.0

                            optimizer_D_DOMAIN.zero_grad()
                            d_domain1.backward()
                            optimizer_D_DOMAIN.step()
                        '''
                        
                        # =================================================================================== #
                        #                                 4. Miscellaneous                                    #
                        # =================================================================================== #
                        # losses_G.update(g_loss.data.cpu().numpy(), config.train['batch_size'])
                        losses_Intrinsic.update(intrinsic_loss.data.cpu().numpy(), config.train['batch_size'])
                        losses_Style.update(style_loss.data.cpu().numpy(), config.train['batch_size'])
                        losses_Integrator.update(integrator_loss.data.cpu().numpy(), config.train['batch_size'])
                        losses_D_X.update(d_loss_x.data.cpu().numpy(), config.train['batch_size'])
                        losses_D_Y.update(d_loss_y.data.cpu().numpy(), config.train['batch_size'])
                        # losses_D_DOMAIN.update(d_domain.data.cpu().numpy(), config.train['batch_size'])
                        
                        # SummaryWriter
                        # writer_loss_G_steps.add_scalar('steps/loss_G', losses_G.avg, i)
                        writer_loss_Intrinsic_steps.add_scalar('steps/loss_Intrinsic', losses_Intrinsic.avg, i)
                        writer_loss_Style_steps.add_scalar('steps/loss_Style', losses_Style.avg, i)
                        writer_loss_Integrator_steps.add_scalar('steps/loss_G', losses_Integrator.avg, i)
                        writer_loss_D_X_steps.add_scalar('steps/loss_D_' + src, losses_D_X.avg, i)
                        writer_loss_D_Y_steps.add_scalar('steps/loss_D_' + tgt, losses_D_Y.avg, i)
                        writer_loss_D_DOMAIN_steps.add_scalar('steps/loss_D_DOMAIN', losses_D_DOMAIN.avg, i)
                        
                        # Determine approximate time left
                        batch_time.update(time.time() - end_time)
                        end_time = time.time()
                        batches_done = epoch * len(trainLoader) + i
                        batches_left = config.train['epochs'] * len(trainLoader) - batches_done
                        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                        prev_time = time.time()
                        
                        bar.suffix = 'Epoch/Step: {epoch}/{step} | LR_G: {lr_G:.8f} | LR_D: {lr_D:.8f}' \
                                     '\n' \
                                     'Loss_intrinsic: {loss_intrinsic:.6f} | Loss_style: {loss_style:.6f} | Loss_integrator: {loss_integrator:.6f} | G_Recon: {g_recon:.6f} | G_Cyc: {g_cycle:.6f} | G_Adv: {g_adv:.6f} | G_Cam: {g_cam:.6f} ' \
                                     '| G_Cls: {g_cls:.6f} | G_Intrinsic: {g_intrinsic:.6f}' \
                                     '\n' \
                                     'Loss_D_VIS: {loss_D_VIS:.6f} | Loss_D_NIR: {loss_D_NIR:.6f} | Loss_D_SKETCH: {loss_D_SKETCH:.6f} | Loss_D_DOMAIN: {loss_D_DOMAIN:.6f} | ETA: {time_left}' \
                                     '\n' \
                                     '***************************************************************************'.format(
                            step=i,
                            epoch=epoch,
                            lr_G=lr_G,
                            lr_D=lr_D,
                            # loss_G=losses_G.avg,
                            loss_intrinsic=losses_Intrinsic.avg,
                            loss_style=losses_Style.avg,
                            loss_integrator=losses_Integrator.avg,
                            g_recon=g_recon.data.cpu().numpy(),
                            g_cycle=g_cycle.data.cpu().numpy(),
                            g_adv=g_adv.data.cpu().numpy(),
                            g_cam=g_cam.data.cpu().numpy(),
                            g_cls=g_cls.data.cpu().numpy(),
                            # g_domain=g_domain.data.cpu().numpy(),
                            g_intrinsic=g_intrinsic.data.cpu().numpy(),
                            loss_D_VIS=losses_D_VIS.avg,
                            loss_D_NIR=losses_D_NIR.avg,
                            loss_D_SKETCH=losses_D_SKETCH.avg,
                            loss_D_DOMAIN=g_mmd.data.cpu().numpy(),
                            time_left=time_left,
                        )
                        print(bar.suffix)
                        # step += 1
                        
                        # Save Image
                        if i % count == 0:
                            fake_y_single = x_real.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'real_{}_{}_{}.png'.format(src, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = y_real.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'real_{}_{}_{}.png'.format(tgt, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = x_rec.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'recon_{}_{}_{}.png'.format(src, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = x_recon.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'cycle_{}_{}_{}.png'.format(src, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = xy.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = '{}2{}_{}_{}.png'.format(src, tgt, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            # heatmap_single = xy_heatmap.detach().cpu().numpy()[0]
                            # heatmap_single = cam(tensor2numpy(heatmap_single), 112)
                            # heatmap_single = heatmap_single * 0.5 + fake_y_single * 0.5
                            # heatmap_single_name = '{}2{}_heatmap_{}_{}.png'.format(src, tgt, epoch, i // count)
                            # save_image_single(heatmap_single, config.train['out'] + heatmap_single_name)
                            
                            fake_y_single = y_rec.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'recon_{}_{}_{}.png'.format(tgt, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = y_recon.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'cycle_{}_{}_{}.png'.format(tgt, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = yx.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = '{}2{}_{}_{}.png'.format(tgt, src, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            # heatmap_single = yx_heatmap.detach().cpu().numpy()[0]
                            # heatmap_single = cam(tensor2numpy(heatmap_single), 112)
                            # heatmap_single = heatmap_single * 0.5 + fake_y_single * 0.5
                            # heatmap_single_name = '{}2{}_heatmap_{}_{}.png'.format(tgt, src, epoch, i // count)
                            # save_image_single(heatmap_single, config.train['out'] + heatmap_single_name)
            
            # SummaryWriter
            # writer_loss_G_epochs.add_scalar('epochs/loss_G', losses_G.avg, epoch)
            writer_loss_Integrator_epochs.add_scalar('epochs/loss_Integrator', losses_Integrator.avg, epoch)
            writer_loss_Intrinsic_epochs.add_scalar('epochs/loss_Intrinsic', losses_Intrinsic.avg, epoch)
            writer_loss_Style_epochs.add_scalar('epochs/loss_Style', losses_Style.avg, epoch)
            writer_loss_D_SKETCH_epochs.add_scalar('epochs/loss_D_SKETCH', losses_D_SKETCH.avg, epoch)
            writer_loss_D_VIS_epochs.add_scalar('epochs/loss_D_VIS', losses_D_VIS.avg, epoch)
            writer_loss_D_NIR_epochs.add_scalar('epochs/loss_D_NIR', losses_D_NIR.avg, epoch)
            writer_loss_D_DOMAIN_epochs.add_scalar('epochs/loss_D_DOMAIN', losses_D_DOMAIN.avg, epoch)
            
            # lr_schedule_G.step()
            lr_schedule_Intrinsic.step()
            lr_schedule_Style.step()
            lr_schedule_Integrator.step()
            lr_schedule_D_VIS.step()
            lr_schedule_D_NIR.step()
            lr_schedule_D_SKETCH.step()
            lr_schedule_D_DOMAIN.step()
            if epoch % 10 == 0:
                date = '20190924'
                
                torch.save({
                    'state_dict_Intrinsic_Encoder': Intrinsic_Encoder.state_dict(),
                    'epoch_Intrinsic_Encoder': epoch,
                }, os.path.join(config.train['checkpoint'], 'Intrinsic_Encoder_' + date + '.pth'))
                
                torch.save({
                    'state_dict_Style_Encoder': Style_Encoder.state_dict(),
                    'epoch_Style_Encoder': epoch,
                }, os.path.join(config.train['checkpoint'], 'Style_Encoder_' + date + '.pth'))
                torch.save({
                    'state_dict_Integrator': Integrator.state_dict(),
                    'epoch_Integrator': epoch,
                }, os.path.join(config.train['checkpoint'], 'Integrator_' + date + '.pth'))
                torch.save({
                    'state_dict_D_VIS': D_VIS.state_dict(),
                    'epoch_D_VIS': epoch
                }, os.path.join(config.train['checkpoint'], 'D_VIS_' + date + '.pth'))
                torch.save({
                    'state_dict_D_NIR': D_NIR.state_dict(),
                    'epoch_D_NIR': epoch
                }, os.path.join(config.train['checkpoint'], 'D_NIR_' + date + '.pth'))
                torch.save({
                    'state_dict_D_SKETCH': D_SKETCH.state_dict(),
                    'epoch_D_SKETCH': epoch
                }, os.path.join(config.train['checkpoint'], 'D_SKETCH_' + date + '.pth'))
                # torch.save({
                #     'state_dict_D_DOMAIN': D_DOMAIN.state_dict(),
                #     'epoch_D_DOMAIN': epoch
                # }, os.path.join(config.train['checkpoint'], 'D_DOMAIN_' + date + '.pth'))
                # torch.save(optimizer_G.state_dict(),
                #            os.path.join(config.train['checkpoint'], 'optimizer_G_' + date + '.pth'))
                torch.save(optimizer_Intrinsic.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_Intrinsic_' + date + '.pth'))
                torch.save(optimizer_Style.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_Style_' + date + '.pth'))
                torch.save(optimizer_Integrator.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_Integrator_' + date + '.pth'))
                torch.save(optimizer_D_VIS.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_D_VIS_' + date + '.pth'))
                torch.save(optimizer_D_NIR.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_D_NIR_' + date + '.pth'))
                torch.save(optimizer_D_SKETCH.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_D_SKETCH_' + date + '.pth'))
                # torch.save(optimizer_D_DOMAIN.state_dict(),
                #            os.path.join(config.train['checkpoint'], 'optimizer_D_DOMAIN_' + date + '.pth'))
