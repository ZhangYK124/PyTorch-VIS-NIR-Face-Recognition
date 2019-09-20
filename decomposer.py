import time
import numpy as np
import torch
import random
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
    optimizer_G = torch.optim.Adam(itertools.chain(Style_Encoder.parameters(), Intrinsic_Encoder.parameters(), Integrator.parameters()),
                                   lr=config.train['lr_G'],
                                   betas=(config.train['beta1_G'], config.train['beta2_G']),
                                   weight_decay=config.train['weight_decay_G'])
    
    if config.train['resume_Style_Encoder']:
        checkpoint = torch.load(config.train['resume_Style_Encoder'])
        Style_Encoder.load_state_dict(checkpoint['state_dict_Style_Encoder'])
        start_epoch = checkpoint['epoch_Style_Encoder']
        
        checkpoint = torch.load(config.train['resume_Intrinsic_Encoder'])
        Intrinsic_Encoder.load_state_dict(checkpoint['state_dict_Intrinsic_Encoder'])
        start_epoch = checkpoint['epoch_Intrinsic_Encoder']
        
        checkpoint = torch.load(config.train['resume_Integrator'])
        Integrator.load_state_dict(checkpoint['state_dict_Integrator'])
        start_epoch = checkpoint['epoch_Integrator']
        optim_checkpoint = torch.load(config.train['resume_optim_G'])
        optimizer_G.load_state_dict(optim_checkpoint)
        for state in optimizer_G.state.values():
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
        
        start_epoch = checkpoint['epoch_D_DOMAIN']
        
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
    
    # LR Schedulers
    lr_schedule_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(config.train['epochs'],
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
    
    count = int(len(trainLoader) // 1)
    
    vis_buffer = ReplayBuffer()
    nir_buffer = ReplayBuffer()
    sketch_buffer = ReplayBuffer()
    
    print('***************** Save Image Iteration: {} ********************'.format(count))
    writer_loss_G_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_G')
    writer_loss_D_VIS_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D_VIS')
    writer_loss_D_NIR_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D_NIR')
    writer_loss_D_SKETCH_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D_SKETCH')
    writer_loss_D_DOMAIN_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D_DOMAIN')
    
    # Training
    if config.train['if_train']:
        for epoch in range(start_epoch, config.train['epochs']):
            writer_loss_G_steps = SummaryWriter(config.train['logs'] + 'steps/loss_G')
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
            losses_G = AverageMeter()
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
                for src in ['sketch', 'nir', 'vis']:
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
                    
                    for tgt in target_list:
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
                            
                            xy_style, xy_style_logit = Style_Encoder(xy)
                            yx_style, yx_style_logit = Style_Encoder(yx)
                            
                            x_recon, x_recon_heatmap, x_recon_cam_logit = Integrator(xy_intrinsic, yx_style)
                            y_recon, y_recon_heatmap, y_recon_cam_logit = Integrator(yx_intrinsic, xy_style)
                            
                            # reconstruction loss
                            g_recon = l1_loss(x_real, x_rec) + l1_loss(y_real, y_rec)
                            
                            # cycle consistency loss
                            g_cycle = l1_loss(x_real, x_recon) + l1_loss(y_real, y_recon)
                            
                            # # mmd loss
                            # g_mmd = mmd(x_intrinsic, y_intrinsic) + mmd(xy_intrinsic, yx_intrinsic) + mmd(x_intrinsic, yx_intrinsic) + mmd(
                            #     y_intrinsic, xy_intrinsic)
                            domain_cls1 = D_DOMAIN(x_intrinsic, y_intrinsic)
                            domain_cls2 = D_DOMAIN(xy_intrinsic, yx_intrinsic)
                            domain_cls3 = D_DOMAIN(x_intrinsic, yx_intrinsic)
                            domain_cls4 = D_DOMAIN(y_intrinsic, xy_intrinsic)
                            g_domain = mse_loss(domain_cls1, torch.ones_like(domain_cls1).cuda()) \
                                       + mse_loss(domain_cls2, torch.ones_like(domain_cls2).cuda()) \
                                       + mse_loss(domain_cls2, torch.ones_like(domain_cls2).cuda()) \
                                       + mse_loss(domain_cls2, torch.ones_like(domain_cls2).cuda())
                            
                            # adv loss & cam loss
                            x_rec_adv, x_rec_adv_cam_logit, _ = D_X(x_rec)
                            y_rec_adv, y_rec_adv_cam_logit, _ = D_Y(y_rec)
                            
                            xy_adv, xy_adv_cam_logit, _ = D_Y(xy)
                            yx_adv, yx_adv_cam_logit, _ = D_X(yx)
                            
                            g_adv = mse_loss(x_rec_adv, torch.ones_like(x_rec_adv).cuda()) + mse_loss(y_rec_adv, torch.ones_like(y_rec_adv).cuda())
                            g_cam = mse_loss(x_rec_adv_cam_logit, torch.ones_like(x_rec_adv_cam_logit).cuda()) \
                                    + mse_loss(y_rec_adv_cam_logit, torch.ones_like(y_rec_adv_cam_logit).cuda())
                            
                            # style cls loss
                            x_cls_label = label2onehot(x_style_logit, c_src)
                            y_cls_label = label2onehot(y_style_logit, c_tgt)
                            
                            g_cls = cross_entropy(x_style_logit, x_cls_label) + cross_entropy(y_style_logit, y_cls_label) \
                                    + cross_entropy(xy_style_logit, y_cls_label) + cross_entropy(yx_style_logit, x_cls_label)
                            
                            g_loss = g_recon * 100.0 + g_cycle * 80.0 + g_adv * 50.0 + g_cam * 100.0 + g_cls * 50.0 + g_domain * 20.0
                            
                            optimizer_G.zero_grad()
                            g_loss.backward()
                            optimizer_G.step()
                        
                        # =================================================================================== #
                        #                             2. Train the discriminator                              #
                        # =================================================================================== #
                        x_rec_adv, x_rec_adv_cam_logit, _ = D_X(x_rec.detach())
                        y_rec_adv, y_rec_adv_cam_logit, _ = D_Y(y_rec.detach())
                        
                        xy_adv, xy_adv_cam_logit, _ = D_Y(xy.detach())
                        yx_adv, yx_adv_cam_logit, _ = D_X(yx.detach())
                        
                        x_adv, x_adv_cam_logit, _ = D_X(x_real)
                        y_adv, y_adv_cam_logit, _ = D_Y(y_real)
                        
                        d_X_adv = mse_loss(x_rec_adv, torch.zeros_like(x_rec_adv).cuda()) + mse_loss(yx_adv, torch.zeros_like(yx_adv).cuda()) \
                                  + mse_loss(x_adv, torch.ones_like(x_adv).cuda())
                        
                        d_X_cam = mse_loss(x_rec_adv_cam_logit, torch.zeros_like(x_rec_adv_cam_logit)) \
                                  + mse_loss(yx_adv_cam_logit, torch.zeros_like(yx_adv_cam_logit)) \
                                  + mse_loss(x_adv_cam_logit, torch.ones_like(x_adv_cam_logit))
                        
                        d_Y_adv = mse_loss(y_rec_adv, torch.zeros_like(y_rec_adv).cuda()) + mse_loss(xy_adv, torch.zeros_like(xy_adv).cuda()) \
                                  + mse_loss(y_adv, torch.ones_like(y_adv).cuda())
                        
                        d_Y_cam = mse_loss(y_rec_adv_cam_logit, torch.zeros_like(y_rec_adv_cam_logit)) \
                                  + mse_loss(xy_adv_cam_logit, torch.zeros_like(xy_adv_cam_logit).cuda()) \
                                  + mse_loss(y_adv_cam_logit, torch.ones_like(y_adv_cam_logit))
                        
                        d_loss_x = d_X_adv * 50.0 + d_X_cam * 100.0
                        d_loss_y = d_Y_adv * 50.0 + d_Y_cam * 100.0
                        
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
                        
                        domain_cls1 = D_DOMAIN(x_intrinsic, y_intrinsic)
                        domain_cls2 = D_DOMAIN(xy_intrinsic, yx_intrinsic)
                        domain_cls3 = D_DOMAIN(x_intrinsic, yx_intrinsic)
                        domain_cls4 = D_DOMAIN(y_intrinsic, xy_intrinsic)
                        d_domain = mse_loss(domain_cls1, torch.zeros_like(domain_cls1).cuda()) \
                                   + mse_loss(domain_cls2, torch.zeros_like(domain_cls2).cuda()) \
                                   + mse_loss(domain_cls2, torch.zeros_like(domain_cls2).cuda()) \
                                   + mse_loss(domain_cls2, torch.zeros_like(domain_cls2).cuda())
                        d_domain = d_domain * 1000.0
                        
                        # Backward and optimize.
                        optimizer_D_X.zero_grad()
                        d_loss_x.backward()
                        optimizer_D_X.step()
                        
                        optimizer_D_Y.zero_grad()
                        d_loss_y.backward()
                        optimizer_D_Y.step()
                        
                        optimizer_D_DOMAIN.zero_grad()
                        d_domain.backward()
                        optimizer_D_DOMAIN.step()
                        
                        # =================================================================================== #
                        #                                 4. Miscellaneous                                    #
                        # =================================================================================== #
                        losses_G.update(g_loss.data.cpu().numpy(), config.train['batch_size'])
                        losses_D_X.update(d_loss_x.data.cpu().numpy(), config.train['batch_size'])
                        losses_D_Y.update(d_loss_y.data.cpu().numpy(), config.train['batch_size'])
                        losses_D_DOMAIN.update(d_domain.data.cpu().numpy(), config.train['batch_size'])
                        
                        # SummaryWriter
                        writer_loss_G_steps.add_scalar('steps/loss_G', losses_G.avg, i)
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
                        
                        bar.suffix = 'Epoch/Step: {epoch}/{step} | LR_G: {lr_G:.8f} | LR_D: {lr_D:.8f} | Loss_G: {loss_G:.6f} |' \
                                     '  Loss_D_VIS: {loss_D_VIS:.6f} | Loss_D_NIR: {loss_D_NIR:.6f} | Loss_D_SKETCH: {loss_D_SKETCH:.6f} | Loss_D_DOMAIN: {loss_D_DOMAIN:.6f} | ETA: {time_left}'.format(
                            step=i * 6 + step,
                            epoch=epoch,
                            lr_G=lr_G,
                            lr_D=lr_D,
                            loss_G=losses_G.avg,
                            loss_D_VIS=losses_D_VIS.avg,
                            loss_D_NIR=losses_D_NIR.avg,
                            loss_D_SKETCH=losses_D_SKETCH.avg,
                            loss_D_DOMAIN=losses_D_DOMAIN.avg,
                            time_left=time_left,
                        )
                        print(bar.suffix)
                        step += 1
                        
                        # Save Image
                        if i % count == 0:
                            fake_y_single = xy.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = '{}2{}_{}_{}.png'.format(src, tgt, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = x_rec.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'recon_{}_{}_{}.png'.format(src, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = x_recon.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'cycle_{}_{}_{}.png'.format(src, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            heatmap_single = xy_heatmap.detach().cpu().numpy()[0]
                            heatmap_single = cam(tensor2numpy(heatmap_single), 112)
                            heatmap_single = heatmap_single * 0.5 + fake_y_single * 0.5
                            heatmap_single_name = '{}2{}_heatmap_{}_{}.png'.format(src, tgt, epoch, i // count)
                            save_image_single(heatmap_single, config.train['out'] + heatmap_single_name)
            
            # SummaryWriter
            writer_loss_G_epochs.add_scalar('epochs/loss_G', losses_G.avg, epoch)
            writer_loss_D_SKETCH_epochs.add_scalar('epochs/loss_D_SKETCH', losses_D_SKETCH.avg, epoch)
            writer_loss_D_VIS_epochs.add_scalar('epochs/loss_D_VIS', losses_D_VIS.avg, epoch)
            writer_loss_D_NIR_epochs.add_scalar('epochs/loss_D_NIR', losses_D_NIR.avg, epoch)
            writer_loss_D_DOMAIN_epochs.add_scalar('epochs/loss_D_DOMAIN', losses_D_DOMAIN.avg, epoch)
            
            lr_schedule_G.step()
            lr_schedule_D_VIS.step()
            lr_schedule_D_NIR.step()
            lr_schedule_D_SKETCH.step()
            lr_schedule_D_DOMAIN.step()
            
            date = '20190919'
            
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
            torch.save({
                'state_dict_D_DOMAIN': D_DOMAIN.state_dict(),
                'epoch_D_DOMAIN': epoch
            }, os.path.join(config.train['checkpoint'], 'D_DOMAIN_' + date + '.pth'))
            torch.save(optimizer_G.state_dict(),
                       os.path.join(config.train['checkpoint'], 'optimizer_G_' + date + '.pth'))
            torch.save(optimizer_D_VIS.state_dict(),
                       os.path.join(config.train['checkpoint'], 'optimizer_D_VIS_' + date + '.pth'))
            torch.save(optimizer_D_NIR.state_dict(),
                       os.path.join(config.train['checkpoint'], 'optimizer_D_NIR_' + date + '.pth'))
            torch.save(optimizer_D_SKETCH.state_dict(),
                       os.path.join(config.train['checkpoint'], 'optimizer_D_SKETCH_' + date + '.pth'))
            torch.save(optimizer_D_DOMAIN.state_dict(),
                       os.path.join(config.train['checkpoint'], 'optimizer_D_DOMAIN_' + date + '.pth'))
