import time
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
import config_progresive as config
from DataLoader_Star import Dataset

from model.model_decomposer_progressive import Discriminator, Generator


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
    if not os.path.exists(config.train['logs']):
        os.mkdir(config.train['logs'])
    if not os.path.exists(config.train['checkpoint']):
        os.mkdir(config.train['checkpoint'])
    if not os.path.exists(config.train['out']):
        os.mkdir(config.train['out'])
    
    if config.train['random_seed'] is None:
        config.train['random_seed'] = random.randint(1, 10000)
    print('Random seed: ', config.train['random_seed'])
    random.seed(config.train['random_seed'])
    torch.manual_seed(config.train['random_seed'])
    
    if config.train['cuda']:
        torch.cuda.manual_seed_all(config.train['random_seed'])
    
    # Dataloader
    trainset = Dataset()
    trainLoader = data.DataLoader(trainset, batch_size=config.train['batch_size'], shuffle=True,
                                  num_workers=config.train['num_workers'])
    
    # Model
    DG_VIS = Discriminator(repeat_num=6)
    DG_NIR = Discriminator(repeat_num=6)
    DG_SKETCH = Discriminator(repeat_num=6)
    DL_VIS = Discriminator(repeat_num=4)
    DL_NIR = Discriminator(repeat_num=4)
    DL_SKETCH = Discriminator(repeat_num=4)
    G = Generator()
    
    # Losses
    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    cross_entropy = torch.nn.CrossEntropyLoss()
    
    # Optimizer
    
    optimizer_D_VIS = torch.optim.Adam(itertools.chain(DG_VIS.parameters(), DL_VIS.parameters()), lr=config.train['lr_D'],
                                       betas=(config.train['beta1_D'], config.train['beta2_D']),
                                       weight_decay=config.train['weight_decay_D'])
    optimizer_D_NIR = torch.optim.Adam(itertools.chain(DG_NIR.parameters(), DL_NIR.parameters()), lr=config.train['lr_D'],
                                       betas=(config.train['beta1_D'], config.train['beta2_D']),
                                       weight_decay=config.train['weight_decay_D'])
    optimizer_D_SKETCH = torch.optim.Adam(itertools.chain(DG_SKETCH.parameters(), DL_SKETCH.parameters()), lr=config.train['lr_D'],
                                          betas=(config.train['beta1_D'], config.train['beta2_D']),
                                          weight_decay=config.train['weight_decay_D'])
    
    # optimizer_DL_VIS = torch.optim.Adam(DL_VIS.parameters(), lr=config.train['lr_D'],
    #                                    betas=(config.train['beta1_D'], config.train['beta2_D']),
    #                                    weight_decay=config.train['weight_decay_D'])
    # optimizer_DL_NIR = torch.optim.Adam(DL_NIR.parameters(), lr=config.train['lr_D'],
    #                                    betas=(config.train['beta1_D'], config.train['beta2_D']),
    #                                    weight_decay=config.train['weight_decay_D'])
    # optimizer_DL_SKETCH = torch.optim.Adam(DL_SKETCH.parameters(), lr=config.train['lr_D'],
    #                                       betas=(config.train['beta1_D'], config.train['beta2_D']),
    #                                       weight_decay=config.train['weight_decay_D'])
    
    optimizer_G_VIS = torch.optim.Adam(itertools.chain(G.Encoder.parameters(),G.Decoder_VIS.parameters()), lr=config.train['lr_G'],
                                   betas=(config.train['beta1_G'], config.train['beta2_G']),
                                   weight_decay=config.train['weight_decay_G'])

    optimizer_G_NIR = torch.optim.Adam(itertools.chain(G.Encoder.parameters(),G.Decoder_NIR.parameters()), lr=config.train['lr_G'],
                                   betas=(config.train['beta1_G'], config.train['beta2_G']),
                                   weight_decay=config.train['weight_decay_G'])

    optimizer_G_SKETCH = torch.optim.Adam(itertools.chain(G.Encoder.parameters(),G.Decoder_SKETCH.parameters()), lr=config.train['lr_G'],
                                   betas=(config.train['beta1_G'], config.train['beta2_G']),
                                   weight_decay=config.train['weight_decay_G'])
    
    # optimizer_D = torch.optim.Adam(itertools.chain(DG_VIS.parameters(),
    #                                                DG_NIR.parameters(),
    #                                                DG_SKETCH.parameters(),
    #                                                DL_VIS.parameters(),
    #                                                DL_NIR.parameters(),
    #                                                DL_SKETCH.parameters()),
    #                                lr=config.train['lr_D'],betas=(config.train['beta1_D'],config.train['beta2_D']),
    #                                weight_decay=config.train['weight_decay_D'])
    
    if config.train['resume_G']:
        checkpoint = torch.load(config.train['resume_G'])
        G.load_state_dict(checkpoint['state_dict_G'])
        start_epoch = checkpoint['epoch_G']
        
        optim_checkpoint = torch.load(config.train['resume_optim_G_VIS'])
        optimizer_G_VIS.load_state_dict(optim_checkpoint)
        for state in optimizer_G_VIS.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        optim_checkpoint = torch.load(config.train['resume_optim_G_NIR'])
        optimizer_G_NIR.load_state_dict(optim_checkpoint)
        for state in optimizer_G_NIR.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
                    
                    
        optim_checkpoint = torch.load(config.train['resume_optim_G_SKETCH'])
        optimizer_G_SKETCH.load_state_dict(optim_checkpoint)
        for state in optimizer_G_SKETCH.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    else:
        G.apply(weights_init)
        start_epoch = 0
    
    if config.train['resume_DG_VIS']:
        checkpoint = torch.load(config.train['resume_DG_VIS'])
        DG_VIS.load_state_dict(checkpoint['state_dict_DG_VIS'])
        
        checkpoint = torch.load(config.train['resume_DL_VIS'])
        DL_VIS.load_state_dict(checkpoint['state_dict_DL_VIS'])

        checkpoint = torch.load(config.train['resume_DG_NIR'])
        DG_NIR.load_state_dict(checkpoint['state_dict_DG_NIR'])

        checkpoint = torch.load(config.train['resume_DL_NIR'])
        DL_NIR.load_state_dict(checkpoint['state_dict_DL_NIR'])
        
        checkpoint = torch.load(config.train['resume_DG_SKETCH'])
        DG_SKETCH.load_state_dict(checkpoint['state_dict_DG_SKETCH'])

        checkpoint = torch.load(config.train['resume_DL_SKETCH'])
        DL_SKETCH.load_state_dict(checkpoint['state_dict_DL_SKETCH'])
        start_epoch = checkpoint['epoch_DL_SKETCH'] + 1

        optim_checkpoint = torch.load(config.train['resume_optim_D_VIS'])
        optimizer_D_VIS.load_state_dict(optim_checkpoint)
        for state in optimizer_D_VIS.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        optim_checkpoint = torch.load(config.train['resume_optim_D_NIR'])
        optimizer_D_NIR.load_state_dict(optim_checkpoint)
        for state in optimizer_D_NIR.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        optim_checkpoint = torch.load(config.train['resume_optim_D_SKETCH'])
        optimizer_D_SKETCH.load_state_dict(optim_checkpoint)
        
        for state in optimizer_D_SKETCH.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    else:
        DG_VIS.apply(weights_init)
        DG_NIR.apply(weights_init)
        DG_SKETCH.apply(weights_init)
        DL_VIS.apply(weights_init)
        DL_NIR.apply(weights_init)
        DL_SKETCH.apply(weights_init)
        start_epoch = 0
    
    if config.train['cuda']:
        DG_VIS.cuda()
        DG_NIR.cuda()
        DG_SKETCH.cuda()
        DL_VIS.cuda()
        DL_NIR.cuda()
        DL_SKETCH.cuda()
        G.cuda()
        l1_loss.cuda()
        mse_loss.cuda()
        cross_entropy.cuda()
    
    # LR Schedulers
    lr_schedule_G_VIS = torch.optim.lr_scheduler.LambdaLR(optimizer_G_VIS, lr_lambda=LambdaLR(config.train['epochs'],
                                                                                      config.train['lr_G_decay_rate'],
                                                                                      config.train['lr_G_decay_epoch'],
                                                                                      config.train['lr_G']).step)
    lr_schedule_G_NIR = torch.optim.lr_scheduler.LambdaLR(optimizer_G_NIR, lr_lambda=LambdaLR(config.train['epochs'],
                                                                                      config.train['lr_G_decay_rate'],
                                                                                      config.train['lr_G_decay_epoch'],
                                                                                      config.train['lr_G']).step)
    lr_schedule_G_SKETCH = torch.optim.lr_scheduler.LambdaLR(optimizer_G_SKETCH, lr_lambda=LambdaLR(config.train['epochs'],
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
    
    # lr_schedule_DL_VIS = torch.optim.lr_scheduler.LambdaLR(optimizer_DL_VIS, lr_lambda=LambdaLR(config.train['epochs'],
    #                                                                                       config.train['lr_D_decay_rate'],
    #                                                                                       config.train['lr_D_decay_epoch'],
    #                                                                                       config.train['lr_D']).step)
    # lr_schedule_DL_NIR = torch.optim.lr_scheduler.LambdaLR(optimizer_DL_NIR, lr_lambda=LambdaLR(config.train['epochs'],
    #                                                                                   config.train['lr_D_decay_rate'],
    #                                                                                   config.train['lr_D_decay_epoch'],
    #                                                                                   config.train['lr_D']).step)
    # lr_schedule_DL_SKETCH = torch.optim.lr_scheduler.LambdaLR(optimizer_DL_SKETCH, lr_lambda=LambdaLR(config.train['epochs'],
    #                                                                                   config.train['lr_D_decay_rate'],
    #                                                                                   config.train['lr_D_decay_epoch'],
    #                                                                                   config.train['lr_D']).step)
    
    prev_time = time.time()
    
    count = int(len(trainLoader) // 1)
    
    print('***************** Save Image Iteration: {} ********************'.format(count))
    writer_loss_G_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_G')
    # writer_loss_D_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D')
    writer_loss_DG_VIS_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_DG_VIS')
    writer_loss_DG_NIR_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_DG_NIR')
    writer_loss_DG_SKETCH_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_DG_SKETCH')
    writer_loss_DL_VIS_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_DL_VIS')
    writer_loss_DL_NIR_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_DL_NIR')
    writer_loss_DL_SKETCH_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_DL_SKETCH')
    
    # Training
    if config.train['if_train']:
        for epoch in range(start_epoch, config.train['epochs']):
            writer_loss_G_steps = SummaryWriter(config.train['logs'] + 'steps/loss_G')
            # writer_loss_D_steps = SummaryWriter(config.train['logs'] + 'steps/loss_D')
            writer_loss_DG_VIS_steps = SummaryWriter(config.train['logs'] + 'steps/loss_DG_VIS')
            writer_loss_DG_NIR_steps = SummaryWriter(config.train['logs'] + 'steps/loss_DG_NIR')
            writer_loss_DG_SKETCH_steps = SummaryWriter(config.train['logs'] + 'steps/loss_DG_SKETCH')
            writer_loss_DL_VIS_steps = SummaryWriter(config.train['logs'] + 'steps/loss_DL_VIS')
            writer_loss_DL_NIR_steps = SummaryWriter(config.train['logs'] + 'steps/loss_DL_NIR')
            writer_loss_DL_SKETCH_steps = SummaryWriter(config.train['logs'] + 'steps/loss_DL_SKETCH')
            
            DG_VIS.train()
            DG_NIR.train()
            DG_SKETCH.train()
            DL_VIS.train()
            DL_NIR.train()
            DL_SKETCH.train()
            G.train()
            
            
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses_G = AverageMeter()
            losses_DG_VIS = AverageMeter()
            losses_DG_NIR = AverageMeter()
            losses_DG_SKETCH = AverageMeter()
            losses_DL_VIS = AverageMeter()
            losses_DL_NIR = AverageMeter()
            losses_DL_SKETCH = AverageMeter()
            
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
                src_list = ['vis', 'nir', 'sketch']
                for src in src_list:
                # if True:
                #     src = choice(src_list)
                    if src == 'vis':
                        c_src = 0
                        DG_X = DG_VIS
                        DL_X = DL_VIS
                        optimizer_G_X = optimizer_G_VIS
                        optimizer_D_X = optimizer_D_VIS
                        losses_DG_X = losses_DG_VIS
                        losses_DL_X = losses_DL_VIS
                        writer_loss_DG_X_steps = writer_loss_DG_VIS_steps
                        writer_loss_DL_X_steps = writer_loss_DL_VIS_steps
                    
                    elif src == 'nir':
                        c_src = 1
                        DG_X = DG_NIR
                        DL_X = DL_NIR
                        optimizer_G_X = optimizer_G_NIR
                        optimizer_D_X = optimizer_D_NIR
                        losses_DG_X = losses_DG_NIR
                        losses_DL_X = losses_DL_NIR
                        writer_loss_DG_X_steps = writer_loss_DG_NIR_steps
                        writer_loss_DL_X_steps = writer_loss_DL_NIR_steps
                    
                    elif src == 'sketch':
                        c_src = 2
                        DG_X = DG_SKETCH
                        DL_X = DL_SKETCH
                        optimizer_G_X = optimizer_G_SKETCH
                        optimizer_D_X = optimizer_D_SKETCH
                        losses_DG_X = losses_DG_SKETCH
                        losses_DL_X = losses_DL_SKETCH
                        writer_loss_DG_X_steps = writer_loss_DG_SKETCH_steps
                        writer_loss_DL_X_steps = writer_loss_DL_SKETCH_steps
                    
                    x_real = batch[src]
                    
                    target_list = ['vis', 'nir', 'sketch']
                    while src in target_list:
                        target_list.remove(src)
                    
                    for tgt in target_list:
                    # if True:
                    #     tgt = choice(target_list)
                        if tgt == 'vis':
                            c_tgt = 0
                            DG_Y = DG_VIS
                            DL_Y = DL_VIS
                            optimizer_G_Y = optimizer_G_VIS
                            optimizer_D_Y = optimizer_D_VIS
                            losses_DG_Y = losses_DG_VIS
                            losses_DL_Y = losses_DL_VIS
                            writer_loss_DG_Y_steps = writer_loss_DG_VIS_steps
                            writer_loss_DL_Y_steps = writer_loss_DL_VIS_steps
                        elif tgt == 'nir':
                            c_tgt = 1
                            DG_Y = DG_NIR
                            DL_Y = DL_NIR
                            optimizer_G_Y = optimizer_G_NIR
                            optimizer_D_Y = optimizer_D_NIR
                            losses_DG_Y = losses_DG_NIR
                            losses_DL_Y = losses_DL_NIR
                            writer_loss_DG_Y_steps = writer_loss_DG_NIR_steps
                            writer_loss_DL_Y_steps = writer_loss_DL_NIR_steps
                        elif tgt == 'sketch':
                            c_tgt = 2
                            DG_Y = DG_SKETCH
                            DL_Y = DL_SKETCH
                            optimizer_G_Y = optimizer_G_SKETCH
                            optimizer_D_Y = optimizer_D_SKETCH
                            losses_DG_Y = losses_DG_SKETCH
                            losses_DL_Y = losses_DL_SKETCH
                            writer_loss_DG_Y_steps = writer_loss_DG_SKETCH_steps
                            writer_loss_DL_Y_steps = writer_loss_DL_SKETCH_steps
                        y_real = batch[tgt]
                    
                        step += 1
                        
                        data_time.update(time.time() - end_time)
                        
                        # =================================================================================== #
                        #                               3. Train the generator                                #
                        # =================================================================================== #
                        if (i + 0) % config.train['generator_steps'] == 0:
                            y_fake = G(x_real, c_tgt)
                            g_adv_g = DG_Y(y_fake.detach())
                            g_adv_l = DL_Y(y_fake.detach())
                            
                            g_loss_src_y_fake = mse_loss(g_adv_g, torch.ones_like(g_adv_g)) \
                                                + mse_loss(g_adv_l, torch.ones_like(g_adv_l))

                            optimizer_G_Y.zero_grad()
                            g_loss_src_y_fake.backward()
                            optimizer_G_Y.step()
                            
                            # reconstruct to the original domain.
                            x_rec = G(y_fake, c_src)
                            g_loss_rec = l1_loss(x_rec, x_real) * config.train['lambda_rec']
                            
                            g_loss = g_loss_src_y_fake + config.train['lambda_rec'] * g_loss_rec
                            
                            optimizer_G_X.zero_grad()
                            g_loss_rec.backward()
                            optimizer_G_X.step()
                        
                        # =================================================================================== #
                        #                             3. Train the discriminator                              #
                        # =================================================================================== #
                        
                        # **************      real      **************
                        dy_adv_g = DG_Y(y_real.detach())
                        dy_adv_l = DL_Y(y_real.detach())
                        dy_adv_real = mse_loss(dy_adv_g, torch.ones_like(dy_adv_g)) \
                                      + mse_loss(dy_adv_l, torch.ones_like(dy_adv_l))
                        dy_g = mse_loss(dy_adv_g, torch.ones_like(dy_adv_g))
                        dy_l = mse_loss(dy_adv_l, torch.ones_like(dy_adv_l))
                        
                        optimizer_D_Y.zero_grad()
                        dy_adv_real.backward()
                        optimizer_D_Y.zero_grad()
                        
                        # **************      fake      **************
                        dy_adv_g = DG_Y(y_fake.detach())
                        dy_adv_l = DL_Y(y_fake.detach())
                        dy_adv_fake = mse_loss(dy_adv_g, torch.zeros_like(dy_adv_g)) \
                                      + mse_loss(dy_adv_l, torch.zeros_like(dy_adv_l))
                        dy_g += mse_loss(dy_adv_g, torch.zeros_like(dy_adv_g))
                        dy_l += mse_loss(dy_adv_l, torch.zeros_like(dy_adv_l))
                        
                        optimizer_D_Y.zero_grad()
                        dy_adv_fake.backward()
                        optimizer_D_Y.zero_grad()
                        
                        # =================================================================================== #
                        #                                 4. Miscellaneous                                    #
                        # =================================================================================== #
                        losses_G.update(g_loss.data.cpu().numpy(), config.train['batch_size'])
                        losses_DG_Y.update(dy_g.data.cpu().numpy(), config.train['batch_size'])
                        losses_DL_Y.update(dy_l.data.cpu().numpy(), config.train['batch_size'])
                        
                        # SummaryWriter
                        writer_loss_G_steps.add_scalar('steps/loss_G', losses_G.avg, i)
                        writer_loss_DG_Y_steps.add_scalar('steps/loss_DG_' + tgt.upper(), losses_DG_Y.avg, i)
                        writer_loss_DL_Y_steps.add_scalar('steps/loss_DL_' + tgt.upper(), losses_DL_Y.avg, i)
                        
                        # Determine approximate time left
                        batch_time.update(time.time() - end_time)
                        end_time = time.time()
                        batches_done = epoch * len(trainLoader) + i
                        batches_left = config.train['epochs'] * len(trainLoader) - batches_done
                        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                        prev_time = time.time()
                        
                        bar.suffix = 'Epoch/Step: {epoch}/{step} | LR_G: {lr_G:.6f} | LR_D: {lr_D:.6f} | L_G: {loss_G:.4f} ' \
                                     '| L_DG_VIS: {loss_DG_VIS:.4f} | L_DL_VIS: {loss_DL_VIS:.4f} ' \
                                     '| L_DL_NIR: {loss_DL_NIR:.4f} | L_DL_NIR: {loss_DL_NIR:.4f} ' \
                                     '| L_DL_SKETCH: {loss_DL_SKETCH:.4f} | L_DL_SKETCH: {loss_DL_SKETCH:.4f} | ETA: {time_left}'.format(
                            step=i * 6 + step,
                            epoch=epoch,
                            lr_G=lr_G,
                            lr_D=lr_D,
                            loss_G=losses_G.avg,
                            loss_DG_VIS=losses_DG_VIS.avg,
                            loss_DL_VIS=losses_DL_VIS.avg,
                            loss_DG_NIR=losses_DG_NIR.avg,
                            loss_DL_NIR=losses_DL_NIR.avg,
                            loss_DG_SKETCH=losses_DG_SKETCH.avg,
                            loss_DL_SKETCH=losses_DL_SKETCH.avg,
                            time_left=time_left,
                        )
                        print(bar.suffix)
                        
                        # Save Image
                        if i % count == 0:
                            fake_y_single = x_real.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'real_{}_{}_{}.png'.format(src, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = y_fake.detach().cpu().numpy()[0]
                            fake_y_single_name = '{}2{}_{}_{}.png'.format(src, tgt, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
                            
                            fake_y_single = x_rec.detach().cpu().numpy()[0]
                            fake_y_single = tensor2numpy(fake_y_single)
                            fake_y_single_name = 'recon_{}_{}_{}.png'.format(src, epoch, i // count)
                            save_image_single(fake_y_single, config.train['out'] + fake_y_single_name)
            
            # SummaryWriter
            writer_loss_G_epochs.add_scalar('epochs/loss_G', losses_G.avg, epoch)
            writer_loss_DG_VIS_epochs.add_scalar('epochs/loss_DG_VIS', losses_DG_VIS.avg, epoch)
            writer_loss_DL_VIS_epochs.add_scalar('epochs/loss_DL_VIS', losses_DL_VIS.avg, epoch)
            writer_loss_DG_NIR_epochs.add_scalar('epochs/loss_DG_NIR', losses_DG_NIR.avg, epoch)
            writer_loss_DL_NIR_epochs.add_scalar('epochs/loss_DL_NIR', losses_DL_NIR.avg, epoch)
            writer_loss_DG_SKETCH_epochs.add_scalar('epochs/loss_DG_SKETCH', losses_DG_SKETCH.avg, epoch)
            writer_loss_DL_SKETCH_epochs.add_scalar('epochs/loss_DL_SKETCH', losses_DL_SKETCH.avg, epoch)
            
            lr_schedule_G_VIS.step()
            lr_schedule_G_NIR.step()
            lr_schedule_G_SKETCH.step()
            lr_schedule_D_VIS.step()
            lr_schedule_D_NIR.step()
            lr_schedule_D_SKETCH.step()
            
            date = '20190929'
            
            if epoch % config.train['save_model'] == 0:
                torch.save({
                    'state_dict_G': G.state_dict(),
                    'epoch_G': epoch,
                }, os.path.join(config.train['checkpoint'], 'G_' + date + '.pth'))
            
                torch.save({
                    'state_dict_DG_VIS': DG_VIS.state_dict(),
                    'epoch_DG_VIS': epoch
                }, os.path.join(config.train['checkpoint'], 'DG_VIS_' + date + '.pth'))
                torch.save({
                    'state_dict_DL_VIS': DL_VIS.state_dict(),
                    'epoch_DL_VIS': epoch
                }, os.path.join(config.train['checkpoint'], 'DL_VIS_' + date + '.pth'))
                
                torch.save({
                    'state_dict_DG_NIR': DG_NIR.state_dict(),
                    'epoch_DG_NIR': epoch
                }, os.path.join(config.train['checkpoint'], 'DG_NIR_' + date + '.pth'))
                torch.save({
                    'state_dict_DL_NIR': DL_NIR.state_dict(),
                    'epoch_DL_NIR': epoch
                }, os.path.join(config.train['checkpoint'], 'DL_NIR_' + date + '.pth'))
                
                torch.save({
                    'state_dict_DG_SKETCH': DG_SKETCH.state_dict(),
                    'epoch_DG_SKETCH': epoch
                }, os.path.join(config.train['checkpoint'], 'DG_SKETCH_' + date + '.pth'))
                torch.save({
                    'state_dict_DL_SKETCH': DL_SKETCH.state_dict(),
                    'epoch_DL_SKETCH': epoch
                }, os.path.join(config.train['checkpoint'], 'DL_SKETCH_' + date + '.pth'))

                torch.save(optimizer_G_VIS.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_G_VIS_' + date + '.pth'))
                torch.save(optimizer_G_NIR.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_G_NIR_' + date + '.pth'))
                torch.save(optimizer_G_SKETCH.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_G_SKETCH_' + date + '.pth'))
                torch.save(optimizer_D_VIS.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_D_VIS_' + date + '.pth'))
                torch.save(optimizer_D_NIR.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_D_NIR_' + date + '.pth'))
                torch.save(optimizer_D_SKETCH.state_dict(),
                           os.path.join(config.train['checkpoint'], 'optimizer_D_SKETCH_' + date + '.pth'))
