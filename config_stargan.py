# TRAIN configuration
train = {}
train['if_train'] = True
train['saparately'] = False

train['epochs'] = 2000
train['batch_size'] = 16
train['num_workers'] = 1
train['generator_steps'] = 3

train['lambda_rec'] = 10.0
train['lambda_gauss'] = 0.0001
train['lambda_cls'] = 1.0

train['lr_D'] = 8e-3
train['lr_D_decay_epoch'] = 50
train['lr_D_decay_rate'] = 0.998
train['beta1_D'] = 0.5
train['beta2_D'] = 0.999
train['weight_decay_D'] = 1e-5

train['lr_G'] = 5e-3
train['lr_G_decay_epoch'] = 50
train['lr_G_decay_rate'] = 0.998
train['beta1_G'] = 0.5
train['beta2_G'] = 0.999
train['weight_decay_G'] = 1e-5

train['resume_G'] = None
train['resume_D'] = None

train['resume_optim_G'] = None
train['resume_optim_D'] = None

# train['resume_G_N2V'] = 'checkpoint/G_N2V_20190807.pth'
# train['resume_G_V2N'] = 'checkpoint/G_V2N_20190807.pth'
# train['resume_D_N'] = 'checkpoint/D_N_20190807.pth'
# train['resume_D_V'] = 'checkpoint/D_V_20190807.pth'
#
# # train['resume_optim_G_V2N'] = 'checkpoint/optimizer_G_V2N_20190807.pth'
# # train['resume_optim_G_N2V'] = 'checkpoint/optimizer_G_N2V_20190807.pth'
# train['resume_optim_G'] = 'checkpoint/optimizer_G_20190807.pth'
# train['resume_optim_D_V'] = 'checkpoint/optimizer_D_V_20190807.pth'
# train['resume_optim_D_N'] = 'checkpoint/optimizer_D_N_20190807.pth'

train['checkpoint'] = './checkpoint1/'
train['logs'] = './logs1/'
train['out'] = './out1/'

train['random_seed'] = None

train['cuda'] = True

# Evaluation configuration
eval = {}
eval['if_eval'] = True


