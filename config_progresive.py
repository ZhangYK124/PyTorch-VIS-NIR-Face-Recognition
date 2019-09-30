# TRAIN configuration
import os
train = {}
train['if_train'] = True
train['saparately'] = False

train['epochs'] = 15000
train['batch_size'] = 2
train['num_workers'] = 0
train['generator_steps'] = 1
train['save_model'] = 50

train['lambda_rec'] = 10.0
train['lambda_gauss'] = 0.0001
train['lambda_cls'] = 1.0

train['lr_D'] = 1e-4
train['lr_D_decay_epoch'] = 1000
train['lr_D_decay_rate'] = 0.999
train['beta1_D'] = 0.5
train['beta2_D'] = 0.999
train['weight_decay_D'] = 1e-5

train['lr_G'] = 1e-4
train['lr_G_decay_epoch'] = 1000
train['lr_G_decay_rate'] = 0.999
train['beta1_G'] = 0.5
train['beta2_G'] = 0.999
train['weight_decay_G'] = 1e-5

# train['resume_G'] = None
# train['resume_DG_VIS'] = None


train['checkpoint'] = './checkpoint_star/'
train['logs'] = './logs_star/'
train['out'] = '/media/hyo/文档/VIS-NIR/RESULT/out_star/'

train['resume_G'] = os.path.join(train['checkpoint'] , 'G_20190929.pth')
train['resume_DG_VIS'] = os.path.join(train['checkpoint'] , 'DG_VIS_20190929.pth')
train['resume_DL_VIS'] = os.path.join(train['checkpoint'] , 'DL_VIS_20190929.pth')
train['resume_DG_NIR'] = os.path.join(train['checkpoint'] , 'DG_NIR_20190929.pth')
train['resume_DL_NIR'] = os.path.join(train['checkpoint'] , 'DL_NIR_20190929.pth')
train['resume_DG_SKETCH'] = os.path.join(train['checkpoint'] , 'DG_SKETCH_20190929.pth')
train['resume_DL_SKETCH'] = os.path.join(train['checkpoint'] , 'DL_SKETCH_20190929.pth')

train['resume_optim_G_VIS'] = os.path.join(train['checkpoint'] , 'optimizer_G_VIS_20190929.pth')
train['resume_optim_G_NIR'] = os.path.join(train['checkpoint'] , 'optimizer_G_NIR_20190929.pth')
train['resume_optim_G_SKETCH'] = os.path.join(train['checkpoint'] , 'optimizer_G_SKETCH_20190929.pth')
train['resume_optim_D_VIS'] = os.path.join(train['checkpoint'] , 'optimizer_D_VIS_20190929.pth')
train['resume_optim_D_NIR'] = os.path.join(train['checkpoint'] , 'optimizer_D_NIR_20190929.pth')
train['resume_optim_D_SKETCH'] = os.path.join(train['checkpoint'] , 'optimizer_D_SKETCH_20190929.pth')

train['random_seed'] = None

train['cuda'] = True

# Evaluation configuration
eval = {}
eval['if_eval'] = True


