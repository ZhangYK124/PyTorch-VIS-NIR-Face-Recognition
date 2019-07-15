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

    # Losses
    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    cross_entropy = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer_D_N = torch.optim.Adam(D_N.parameters(), lr=config.train['lr_D_N'],
                                     betas=(config.train['beta1_D_N'], config.train['beta2_D_N']),
                                     weight_decay=config.train['weight_decay_D_N'])
    optimizer_D_V = torch.optim.Adam(D_V.parameters(), lr=config.train['lr_D_V'],
                                     betas=(config.train['beta1_D_V'], config.train['beta2_D_V']),
                                     weight_decay=config.train['weight_decay_D_V'])
    # optimizer_G_V2N = torch.optim.Adam(G_V2N.parameters(), lr=config.train['lr_G_V2N'],
    #                                    betas=(config.train['beta1_G_V2N'], config.train['beta2_G_N2V']),
    #                                    weight_decay=config.train['weight_decay_G_V2N'])
    # optimizer_G_N2V = torch.optim.Adam(G_N2V.parameters(), lr=config.train['lr_G_N2V'],
    #                                    betas=(config.train['beta1_G_N2V'], config.train['beta2_G_N2V']),
    #                                    weight_decay=config.train['weight_decay_G_N2V'])
    optimizer_G = torch.optim.Adam(itertools.chain(G_N2V.parameters(), G_V2N.parameters()), lr=config.train['lr_G'],
                                   betas=(config.train['beta1_G'], config.train['beta2_G']),
                                   weight_decay=config.train['weight_decay_G'])
    
    if config.train['resume_G_N2V']:
        checkpoint = torch.load(config.train['resume_G_N2V'])
        G_N2V.load_state_dict(checkpoint['state_dict_G_N2V'])
        start_epoch = checkpoint['epoch_G_N2V']
        optim_checkpoint = torch.load(config.train['resume_optim_G'])
        optimizer_G.load_state_dict(optim_checkpoint)
    else:
        G_N2V.apply(weights_init)
        start_epoch = 0

    if config.train['resume_G_V2N']:
        checkpoint = torch.load(config.train['resume_G_V2N'])
        G_V2N.load_state_dict(checkpoint['state_dict_G_V2N'])
        start_epoch = checkpoint['epoch_G_V2N']
        optim_checkpoint = torch.load(config.train['resume_optim_G'])
        optimizer_G.load_state_dict(optim_checkpoint)
        for state in optimizer_G.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):
                    state[k] = v.cuda()
    else:
        G_V2N.apply(weights_init)
        start_epoch = 0

    if config.train['resume_D_V']:
        checkpoint = torch.load(config.train['resume_D_V'])
        D_V.load_state_dict(checkpoint['state_dict_D_V'])
        start_epoch = checkpoint['epoch_D_V']

        optim_checkpoint = torch.load(config.train['resume_optim_D_V'])
        optimizer_D_V.load_state_dict(optim_checkpoint)
        
        for state in optimizer_D_V.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):
                    state[k] = v.cuda()
    else:
        D_V.apply(weights_init)
        start_epoch = 0

    if config.train['resume_D_N']:
        checkpoint = torch.load(config.train['resume_D_N'])
        D_N.load_state_dict(checkpoint['state_dict_D_N'])
        start_epoch = checkpoint['epoch_D_N']

        optim_checkpoint = torch.load(config.train['resume_optim_D_N'])
        optimizer_D_N.load_state_dict(optim_checkpoint)
        
        for state in optimizer_D_N.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):
                    state[k] = v.cuda()
    else:
        D_N.apply(weights_init)
        start_epoch = 0
    
    if config.train['cuda']:
        D_V.cuda()
        D_N.cuda()
        G_N2V.cuda()
        G_V2N.cuda()
        l1_loss.cuda()
        mse_loss.cuda()
        cross_entropy.cuda()
    
    # LR Schedulers
    lr_schedule_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(config.train['epochs'],config.train['lr_G_decay_rate'],config.train['lr_G_decay_epoch'],config.train['lr_G']).step)
    lr_schedule_D_V = torch.optim.lr_scheduler.LambdaLR(optimizer_D_V,lr_lambda=LambdaLR(config.train['epochs'],config.train['lr_D_V_decay_rate'],config.train['lr_D_V_decay_epoch'],config.train['lr_D_V']).step)
    lr_schedule_D_N = torch.optim.lr_scheduler.LambdaLR(optimizer_D_N,lr_lambda=LambdaLR(config.train['epochs'],config.train['lr_D_N_decay_rate'],config.train['lr_D_N_decay_epoch'],config.train['lr_D_N']).step)
    
        
    target_real = Variable(torch.cuda.FloatTensor(config.train['batch_size']).fill_(1.0),requires_grad=False)
    target_fake = Variable(torch.cuda.FloatTensor(config.train['batch_size']).fill_(0.0),requires_grad=False)
    
    fake_vis_buffer = ReplayBuffer()
    fake_nir_buffer = ReplayBuffer()
    fake_local_vis_left_eye_buffer = ReplayBuffer()
    fake_local_vis_right_eye_buffer = ReplayBuffer()
    fake_local_nir_left_eye_buffer = ReplayBuffer()
    fake_local_nir_right_eye_buffer = ReplayBuffer()

    prev_time = time.time()

    count = int(len(trainLoader) // 4)
    writer_loss_G_epochs = SummaryWriter('./logs/epochs/loss_G')
    writer_in_loss_epochs = SummaryWriter('./logs/epochs/in_loss')
    writer_cycle_loss_epochs = SummaryWriter('./logs/epochs/cycle_loss')
    writer_loss_D_N_epochs = SummaryWriter('./logs/epochs/loss_D_N')
    writer_loss_D_V_epochs = SummaryWriter('./logs/epochs/loss_D_V')
    
    # Training
    if config.train['if_train']:
        for epoch in range(start_epoch,config.train['epochs']):
            writer_loss_G_steps = SummaryWriter('./logs/steps/loss_G')
            writer_in_loss_steps = SummaryWriter('./logs/steps/in_loss')
            writer_cycle_loss_steps = SummaryWriter('./logs/steps/cycle_loss')
            writer_loss_D_N_steps = SummaryWriter('./logs/steps/loss_D_N')
            writer_loss_D_V_steps = SummaryWriter('./logs/steps/loss_D_V')
            
            D_V.train()
            D_N.train()
            G_V2N.train()
            G_N2V.train()
            
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses_G = AverageMeter()
            cycle_losses = AverageMeter()
            intensity_losses = AverageMeter()
            losses_G_V2N = AverageMeter()
            losses_G_N2V = AverageMeter()
            losses_D_V = AverageMeter()
            losses_D_N = AverageMeter()

            lr_G = LambdaLR(config.train['epochs'], config.train['lr_G_decay_rate'], config.train['lr_G_decay_epoch'],config.train['lr_G']).step(epoch)
            lr_D_V = LambdaLR(config.train['epochs'], config.train['lr_D_V_decay_rate'], config.train['lr_D_V_decay_epoch'],config.train['lr_D_V']).step(epoch)
            lr_D_N = LambdaLR(config.train['epochs'], config.train['lr_D_N_decay_rate'], config.train['lr_D_N_decay_epoch'],config.train['lr_D_N']).step(epoch)
            
            end_time = time.time()
            
            bar = Bar('Processing: ',max=len(trainLoader))
            
            for i,batch in enumerate(trainLoader):
                data_time.update(time.time()-end_time)
                for k in batch:
                    batch[k] = batch[k].cuda()
                real_vis = batch['vis']
                real_nir = batch['nir']
                real_vis_left_eye = batch['vis_left_eye']
                real_vis_right_eye = batch['vis_right_eye']
                real_nir_left_eye = batch['nir_left_eye']
                real_nir_right_eye = batch['nir_right_eye']
                if i % config.train['generator_steps'] == 0:
                    # ################### Train G ###################
                    optimizer_G.zero_grad()
                    
                    # Gan Loss
                    fake_vis, fake_local_vis, real_local_nir,fake_local_vis_left_eye,fake_local_vis_right_eye = G_N2V(real_nir,real_nir_left_eye,real_nir_right_eye)
                    pred_fake_vis = D_V(fake_vis)
                    pred_fake_vis_local_left_eye = D_V(fake_local_vis_left_eye)
                    pred_fake_vis_local_right_eye = D_V(fake_local_vis_right_eye)
                    
                    loss_G_N2V = mse_loss(pred_fake_vis,target_real) + mse_loss(pred_fake_vis_local_left_eye,target_real) + mse_loss(pred_fake_vis_local_right_eye,target_real)
                    
                    fake_nir, fake_local_nir, real_local_vis,fake_local_nir_left_eye,fake_local_nir_right_eye = G_V2N(real_vis,real_vis_left_eye,real_vis_right_eye)
                    pred_fake_nir = D_N(fake_nir)
                    pred_fake_nir_local_left_eye = D_N(fake_local_nir_left_eye)
                    pred_fake_nir_local_right_eye = D_N(fake_local_nir_right_eye)
                    
                    loss_G_V2N = mse_loss(pred_fake_nir,target_real) + mse_loss(pred_fake_nir_local_left_eye,target_real) + mse_loss(pred_fake_nir_local_right_eye,target_real)
                    
                    loss_G_total = loss_G_V2N+loss_G_N2V
                    
                    # Intensity Loss
                    intensity_loss = (l1_loss(rgb2ycbcr(fake_vis),real_nir[:,0,:,:])
                                      + l1_loss(fake_nir[:,0,:,:],rgb2ycbcr(real_vis))
                                      + l1_loss(rgb2ycbcr(fake_local_vis_left_eye),real_nir_left_eye[:,0,:,:])
                                      + l1_loss(rgb2ycbcr(fake_local_nir_right_eye),real_nir_right_eye[:,0,:,:])
                                      + l1_loss(fake_local_nir_left_eye[:,0,:,:],rgb2ycbcr(real_vis_left_eye))
                                      + l1_loss(fake_local_nir_right_eye[:,0,:,:],rgb2ycbcr(real_vis_right_eye))
                                      ) * config.train['lambda_in_loss']
                    
                    
                    # Identity Loss
                    # identity_fake_nir, identity_fake_local_nir, identity_real_local_nir, identity_fake_local_nir_left_eye, identity_fake_local_nir_right_eye = G_V2N(real_nir, real_nir_left_eye, real_nir_right_eye)
                    # identity_fake_vis, identity_fake_local_vis, identity_real_local_vis, identity_fake_local_vis_left_eye, identity_fake_local_vis_right_eye = G_N2V(real_vis,real_vis_left_eye, real_vis_right_eye)
                    # identity_loss_G_V2N = l1_loss(identity_fake_vis,real_vis) + l1_loss(identity_fake_local_vis,identity_real_local_vis)
                    # identity_loss_G_N2V = l1_loss(identity_fake_nir,real_nir) + l1_loss(identity_fake_local_nir,identity_real_local_nir)
                    # identity_loss = (identity_loss_G_V2N + identity_loss_G_N2V)* config.train['lambda_id_loss']
                    # identity_loss = torch.Tensor(np.array([0.])).cuda()
                    
                    # Cycle Loss
                    recovered_nir,recovered_local_nir,_,recovered_local_nir_left_eye,recovered_local_nir_right_eye = G_V2N(fake_vis,fake_local_vis_left_eye,fake_local_vis_right_eye)
                    cycle_loss_NVN = l1_loss(recovered_nir,real_nir) + l1_loss(recovered_local_nir_left_eye,real_nir_left_eye) + l1_loss(recovered_local_nir_right_eye,real_nir_right_eye)
                    
                    recovered_vis,recovered_local_vis,_,recovered_local_vis_left_eye,recovered_local_vis_right_eye = G_N2V(fake_nir,fake_local_nir_left_eye,fake_local_nir_right_eye)
                    cycle_loss_VNV = l1_loss(recovered_vis,real_vis) + l1_loss(recovered_local_vis_left_eye,real_vis_left_eye)+ l1_loss(recovered_local_vis_right_eye,real_vis_right_eye)
                    
                    cycle_loss = (cycle_loss_NVN + cycle_loss_VNV) * config.train['lambda_cyc_loss']
                    
                    # Total Loss
                    # loss_G = loss_G_N2V + loss_G_V2N + cycle_loss  + identity_loss
                    loss_G = loss_G_total  + cycle_loss + intensity_loss
                    
                    loss_G.backward()
                    optimizer_G.step()
                    # #####################################################
                
                
                # ##################### Train D_V #####################
                optimizer_D_V.zero_grad()
                
                # Real Loss
                pred_real_vis = D_V(real_vis)
                pred_real_local_vis_left_eye = D_V(real_vis_left_eye)
                pred_real_local_vis_right_eye = D_V(real_vis_right_eye)
                loss_D_V_real = mse_loss(pred_real_vis,target_real) + mse_loss(pred_real_local_vis_left_eye,target_real) + mse_loss(pred_real_local_vis_right_eye,target_real)
                
                # Fake Loss
                fake_vis = fake_vis_buffer.push_and_pop(fake_vis)
                fake_local_vis_left_eye = fake_local_vis_left_eye_buffer.push_and_pop(fake_local_vis_left_eye)
                fake_local_vis_right_eye = fake_local_vis_right_eye_buffer.push_and_pop(fake_local_vis_right_eye)
                pred_fake_vis = D_V(fake_vis.detach())
                pred_fake_local_vis_left_eye = D_V(fake_local_vis_left_eye.detach())
                pred_fake_local_vis_right_eye = D_V(fake_local_vis_right_eye.detach())
                loss_D_V_fake = mse_loss(pred_fake_vis,target_fake) + mse_loss(pred_fake_local_vis_left_eye,target_fake) + mse_loss(pred_fake_local_vis_right_eye,target_fake)
                
                # Total Loss
                loss_D_V = loss_D_V_real + loss_D_V_fake
                loss_D_V.backward()
                optimizer_D_V.step()
                # #######################################################
                
                # ########################### Train D_N #############################
                optimizer_D_N.zero_grad()
                
                # Real Loss
                pred_real_nir = D_N(real_nir)
                pred_real_local_nir_left_eye = D_N(real_nir_left_eye)
                pred_real_local_nir_right_eye = D_N(real_nir_right_eye)
                loss_D_real = mse_loss(pred_real_nir,target_real) + mse_loss(pred_real_local_nir_left_eye,target_real) + mse_loss(pred_real_local_nir_right_eye,target_real)
                
                # Fake Loss
                fake_nir = fake_nir_buffer.push_and_pop(fake_nir)
                fake_local_nir_left_eye = fake_local_nir_left_eye_buffer.push_and_pop(fake_local_nir_left_eye)
                fake_local_nir_right_eye = fake_local_nir_right_eye_buffer.push_and_pop(fake_local_nir_right_eye)
                pred_fake_nir = D_N(fake_nir.detach())
                pred_fake_local_nir_left_eye = D_N(fake_local_nir_left_eye.detach())
                pred_fake_local_nir_right_eye = D_N(fake_local_nir_right_eye.detach())
                loss_D_fake = mse_loss(pred_fake_nir,target_fake) + mse_loss(pred_fake_local_nir_left_eye,target_fake) + mse_loss(pred_fake_local_nir_right_eye,target_fake)
                
                # Total Loss
                loss_D_N = loss_D_real + loss_D_fake
                loss_D_N.backward()
                optimizer_D_N.step()
                # ##########################################################
                
                # ######################## Plot Progress ###########################
                losses_G.update(loss_G_total.data.cpu().numpy(),config.train['batch_size'])
                cycle_losses.update(cycle_loss.data.cpu().numpy(),config.train['batch_size'])
                intensity_losses.update(intensity_loss.data.cpu().numpy(),config.train['batch_size'])
                losses_D_N.update(loss_D_N.data.cpu().numpy(),config.train['batch_size'])
                losses_D_V.update(loss_D_V.data.cpu().numpy(),config.train['batch_size'])
                
                batch_time.update(time.time()-end_time)
                end_time = time.time()

                # Determine approximate time left
                batches_done = epoch * len(trainLoader) + i
                batches_left = config.train['epochs'] * len(trainLoader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                
                bar.suffix = 'Epoch/Step: {epoch}/{step} | LR_G: {lr_G:.8f} | LR_D_V: {lr_D_V:.8f} | LR_D_N: {lr_D_N:.8f} | ' \
                             'Loss_G: {loss_G:.6f} | Cycle Loss: {cycle_loss:.6f} | In Loss: {in_loss:.6f} | Loss_D_N: {loss_D_N:.6f} | Loss_D_V: {loss_D_V:.6f} | ETA: {time_left}'.format(
                    step=i,
                    epoch=epoch,
                    lr_G=lr_G,
                    lr_D_V=lr_D_V,
                    lr_D_N=lr_D_N,
                    loss_G=losses_G.avg,
                    cycle_loss=cycle_losses.avg,
                    in_loss=intensity_losses.avg,
                    loss_D_N=losses_D_N.avg,
                    loss_D_V=losses_D_V.avg,
                    time_left=time_left,
                )
                print(bar.suffix)
                
                # SummaryWriter
                writer_loss_G_steps.add_scalar('steps/loss_G',losses_G.avg,i)
                writer_cycle_loss_steps.add_scalar('steps/cycle_loss',cycle_losses.avg,i)
                writer_in_loss_steps.add_scalar('steps/in_loss',intensity_losses.avg,i)
                writer_loss_D_N_steps.add_scalar('steps/loss_D_N',losses_D_N.avg,i)
                writer_loss_D_V_steps.add_scalar('steps/loss_D_V',losses_D_V.avg,i)
                
                # Save Image
                if i%count==0:
                    fake_nir_single = fake_nir.detach().cpu().numpy()[0]
                    fake_nir_single_name = 'fake_nir_{}_{}.png'.format(epoch,i//count)
                    save_image_single(fake_nir_single, './out/' + fake_nir_single_name, 112)
    
                    fake_vis_single = fake_vis.detach().cpu().numpy()[0]
                    fake_vis_single_name = 'fake_vis_{}_{}.png'.format(epoch,i//count)
                    save_image_single(fake_vis_single, './out/' + fake_vis_single_name, 112)
    
                    fake_local_vis_single = fake_local_vis.detach().cpu().numpy()[0]
                    fake_local_vis_single_name = 'fake_local_vis_{}_{}.png'.format(epoch,i//count)
                    save_image_single(fake_local_vis_single, './out/' + fake_local_vis_single_name, 112)
    
                    fake_local_nir_single = fake_local_nir.detach().cpu().numpy()[0]
                    fake_local_nir_single_name = 'fake_local_nir_{}_{}.png'.format(epoch,i//count)
                    save_image_single(fake_local_nir_single, './out/' + fake_local_nir_single_name, 112)
                    
                    real_nir_single = real_nir.detach().cpu().numpy()[0]
                    real_nir_single_name = 'real_nir_{}_{}.png'.format(epoch,i//count)
                    save_image_single(real_nir_single,'./out/'+real_nir_single_name,112)
                    
                    real_vis_single = real_vis.detach().cpu().numpy()[0]
                    real_vis_single_name = 'real_vis_{}_{}.png'.format(epoch,i//count)
                    save_image_single(real_vis_single,'./out/'+real_vis_single_name,112)
                    
                    # fake_local_vis_right_eye_single = fake_local_vis_right_eye.detach().cpu().numpy()[0]
                    # fake_local_vis_right_eye_single_name = 'fake_local_vis_right_eye_{}_{}.png'.format(epoch,i//count)
                    # save_image_single(fake_local_vis_right_eye_single,'./out/'+fake_local_vis_right_eye_single_name)

            # SummaryWriter
            writer_loss_G_epochs.add_scalar('epochs/loss_G', losses_G.avg, epoch)
            writer_cycle_loss_epochs.add_scalar('epochs/in_loss', cycle_losses.avg,epoch)
            writer_in_loss_epochs.add_scalar('epochs/cycle_loss', intensity_losses.avg,epoch)
            writer_loss_D_N_epochs.add_scalar('epochs/loss_D_N',losses_D_N.avg,epoch)
            writer_loss_D_V_epochs.add_scalar('epochs/loss_D_V',losses_D_V.avg,epoch)
            
            lr_schedule_G.step()
            lr_schedule_D_N.step()
            lr_schedule_D_V.step()
            
            date = '20190715'
            
            torch.save({
                'state_dict_G_N2V':G_N2V.state_dict(),
                'epoch_G_N2V':epoch,
            },'./checkpoint/G_N2V_'+date+'.pth')
            torch.save({
                'state_dict_G_V2N':G_V2N.state_dict(),
                'epoch_G_V2N':epoch
            },'./checkpoint/G_V2N_'+date+'.pth')
            torch.save({
                'state_dict_D_V':D_V.state_dict(),
                'epoch_D_V':epoch
            },'./checkpoint/D_V_'+date+'.pth')
            torch.save({
                'state_dict_D_N':D_N.state_dict(),
                'epoch_D_N':epoch
            },'./checkpoint/D_N_'+date+'.pth')
            torch.save(optimizer_G.state_dict(),'./checkpoint/optimizer_G_'+date+'.pth')
            torch.save(optimizer_D_V.state_dict(),'./checkpoint/optimizer_D_V_'+date+'.pth')
            torch.save(optimizer_D_N.state_dict(),'./checkpoint/optimizer_D_N_'+date+'.pth')
            
            
            
            
                
                
                
                
                
                
                
                
                
                
            
            
            
        
        
        
        
        
    
