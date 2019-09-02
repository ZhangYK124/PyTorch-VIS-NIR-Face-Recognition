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
import config_stargan as config
from DataLoader_Star import Dataset

from model.model_star import Discriminator, Generator


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
            
def label2onehot(labels,c):
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
    
    if config.train['cuda']:
        torch.cuda.manual_seed_all(config.train['random_seed'])
    
    # Dataloader
    trainset = Dataset()
    trainLoader = data.DataLoader(trainset, batch_size=config.train['batch_size'], shuffle=True,
                                  num_workers=config.train['num_workers'])
    
    # Model
    D_VIS = Discriminator()
    D_NIR = Discriminator()
    D_SKETCH = Discriminator()
    G = Generator()
    
    # Losses
    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    cross_entropy = torch.nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer_D_VIS = torch.optim.Adam(D_VIS.parameters(), lr=config.train['lr_D'],
                                     betas=(config.train['beta1_D'], config.train['beta2_D']),
                                     weight_decay=config.train['weight_decay_D'])
    optimizer_D_NIR = torch.optim.Adam(D_NIR.parameters(), lr=config.train['lr_D'],
                                   betas=(config.train['beta1_D'], config.train['beta2_D']),
                                   weight_decay=config.train['weight_decay_D'])
    optimizer_D_SKETCH = torch.optim.Adam(D_SKETCH.parameters(), lr=config.train['lr_D'],
                                   betas=(config.train['beta1_D'], config.train['beta2_D']),
                                   weight_decay=config.train['weight_decay_D'])
    optimizer_G = torch.optim.Adam(itertools.chain(G.parameters(), G.parameters()), lr=config.train['lr_G'],
                                   betas=(config.train['beta1_G'], config.train['beta2_G']),
                                   weight_decay=config.train['weight_decay_G'])
    
    if config.train['resume_G']:
        checkpoint = torch.load(config.train['resume_G'])
        G.load_state_dict(checkpoint['state_dict_G'])
        start_epoch = checkpoint['epoch_G']
        optim_checkpoint = torch.load(config.train['resume_optim_G'])
        optimizer_G.load_state_dict(optim_checkpoint)
        for state in optimizer_G.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    else:
        G.apply(weights_init)
        start_epoch = 0
    
    if config.train['resume_D']:
        checkpoint = torch.load(config.train['resume_D'])
        D.load_state_dict(checkpoint['state_dict_D'])
        start_epoch = checkpoint['epoch_D']
        
        optim_checkpoint = torch.load(config.train['resume_optim_D'])
        optimizer_D.load_state_dict(optim_checkpoint)
        
        for state in optimizer_D.state.values():
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
        G.cuda()
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

    
    prev_time = time.time()
    
    count = int(len(trainLoader) // 5)
    
    print('***************** Save Image Iteration: {} ********************'.format(count))
    writer_loss_G_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_G')
    writer_loss_D_epochs = SummaryWriter(config.train['logs'] + 'epochs/loss_D')
    
    # Training
    if config.train['if_train']:
        for epoch in range(start_epoch, config.train['epochs']):
            writer_loss_G_steps = SummaryWriter(config.train['logs'] + 'steps/loss_G')
            writer_loss_D_steps = SummaryWriter(config.train['logs'] + 'steps/loss_D')
            
            D_VIS.train()
            D_NIR.train()
            D_SKETCH.train()
            G.train()
            
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses_G = AverageMeter()
            losses_D = AverageMeter()
            
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
                for src in ['vis','nir','sketch']:
                    if src=='vis':
                        c_src = 0
                    elif src=='nir':
                        c_src = 1
                    elif src=='sketch':
                        c_src = 2
                    x_real = batch[src]
                    
                    for tgt in['vis','nir','sketch']:
                        if tgt == 'vis':
                            c_tgt = 0
                            D = D_VIS
                            optimizer_D = optimizer_D_VIS
                        elif tgt == 'nir':
                            c_tgt = 1
                            D = D_NIR
                            optimizer_D = optimizer_D_NIR
                        elif tgt == 'sketch':
                            c_tgt = 2
                            D = D_SKETCH
                            optimizer_D_SKETCH = optimizer_D_SKETCH
                        y_real = batch[tgt]
                        
                        data_time.update(time.time() - end_time)

                        
                        # =================================================================================== #
                        #                             2. Train the discriminator                              #
                        # =================================================================================== #
                        out_src , out_cls = D(x_real)
                        d_loss_src_x_real = mse_loss(out_src,torch.ones_like(out_src).cuda())
                        cls_label = label2onehot(out_cls,c_src)
                        d_loss_cls_x_real = cross_entropy(out_cls,cls_label)
                        
                        out_src , out_cls = D(y_real)
                        d_loss_src_y_real = mse_loss(out_src,torch.ones_like(out_src).cuda())
                        cls_label = label2onehot(out_cls,c_tgt)
                        d_loss_cls_y_real = cross_entropy(out_cls,cls_label)
                        
                        y_fake = G(x_real,c_tgt)
                        out_src,out_cls = D(y_fake.detach())
                        d_loss_src_y_fake = mse_loss(out_src,torch.zeros_like(out_src).cuda())
                        cls_label = label2onehot(out_cls,c_tgt)
                        d_loss_cls_y_fake = cross_entropy(out_cls,cls_label)
                        
                        # x_rec = G(y_fake,c_src)
                        # out_src,out_cls = D(x_rec.detach())
                        # d_loss_src_x_rec = mse_loss(out_src,torch.zeros_like(out_src).cuda())
                        # cls_label = label2onehot(out_cls,c_src)
                        # d_loss_cls_x_rec = cross_entropy(out_cls,cls_label)
                        
                        d_loss = d_loss_src_x_real + d_loss_src_y_real + d_loss_src_y_fake  \
                                 + config.train['lambda_cls'] * (d_loss_cls_x_real + d_loss_cls_y_fake + d_loss_cls_y_real )
                        
                        # Backward and optimize.
                        optimizer_D.zero_grad()
                        d_loss.backward()
                        optimizer_D.step()

                        # =================================================================================== #
                        #                               3. Train the generator                                #
                        # =================================================================================== #
                        if (i + 0) % config.train['generator_steps'] == 0:
                            y_fake = G(x_real, c_tgt)
                            out_src, out_cls = D(y_fake.detach())
                            g_loss_src_y_fake = mse_loss(out_src, torch.ones_like(out_src).cuda())
                            cls_label = label2onehot(out_cls, c_tgt)
                            g_loss_cls_y_fake = cross_entropy(out_cls, cls_label)
    
                            # reconstruct to the original domain.
                            x_rec = G(y_fake, c_src)
                            g_loss_rec = l1_loss(x_rec, x_real)
                            # out_src, out_cls = D(x_rec.detach())
                            # g_loss_src_x_rec = mse_loss(out_src, torch.ones_like(out_src).cuda())
                            # cls_label = label2onehot(out_cls, c_src)
                            # g_loss_cls_x_rec = cross_entropy(out_cls, cls_label)
    
                            g_loss = g_loss_src_y_fake \
                                     + config.train['lambda_cls'] * (g_loss_cls_y_fake) \
                                     + config.train['lambda_rec'] * g_loss_rec
                            optimizer_G.zero_grad()
                            g_loss.backward()
                            optimizer_G.step()

                        # =================================================================================== #
                        #                                 4. Miscellaneous                                    #
                        # =================================================================================== #
                        losses_G.update(g_loss.data.cpu().numpy(),config.train['batch_size'])
                        losses_D.update(d_loss.data.cpu().numpy(),config.train['batch_size'])

                        # SummaryWriter
                        writer_loss_G_steps.add_scalar('steps/loss_G', losses_G.avg, i)
                        writer_loss_D_steps.add_scalar('steps/loss_D', losses_D.avg, i)
                        
                        # Determine approximate time left
                        batch_time.update(time.time() - end_time)
                        end_time = time.time()
                        batches_done = epoch * len(trainLoader) + i
                        batches_left = config.train['epochs'] * len(trainLoader) - batches_done
                        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                        prev_time = time.time()

                        bar.suffix = 'Epoch/Step: {epoch}/{step} | LR_G: {lr_G:.8f} | LR_D: {lr_D:.8f} | Loss_G: {loss_G:.6f} |  Loss_D: {loss_D:.6f} | ETA: {time_left}'.format(
                            step=i*6+step,
                            epoch=epoch,
                            lr_G=lr_G,
                            lr_D=lr_D,
                            loss_G=losses_G.avg,
                            loss_D=losses_D.avg,
                            time_left=time_left,
                        )
                        print(bar.suffix)
                        step += 1
                        
                        # Save Image
                        if i % count == 0:
                            fake_y_single = y_fake.detach().cpu().numpy()[0]
                            fake_y_single_name = '{}2{}_{}_{}.png'.format(src,tgt,epoch,i//count)
                            save_image_single(fake_y_single,config.train['out'] + fake_y_single_name)
             
            # SummaryWriter
            writer_loss_G_epochs.add_scalar('epochs/loss_G', losses_G.avg, epoch)
            writer_loss_D_epochs.add_scalar('epochs/loss_D', losses_D.avg, epoch)
            
            lr_schedule_G.step()
            lr_schedule_D_VIS.step()
            lr_schedule_D_NIR.step()
            lr_schedule_D_SKETCH.step()
            
            date = '20190830'
            
            torch.save({
                'state_dict_G': G.state_dict(),
                'epoch_G': epoch,
            }, os.path.join(config.train['checkpoint'], 'G_' + date + '.pth'))
            torch.save({
                'state_dict_D': D.state_dict(),
                'epoch_D': epoch
            }, os.path.join(config.train['checkpoint'], 'D_' + date + '.pth'))
            torch.save(optimizer_G.state_dict(),
                       os.path.join(config.train['checkpoint'], 'optimizer_G_' + date + '.pth'))
            torch.save(optimizer_D.state_dict(),
                       os.path.join(config.train['checkpoint'], 'optimizer_D_' + date + '.pth'))

