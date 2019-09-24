# TRAIN configuration
train = {}
train['if_train'] = True
train['saparately'] = False

train['epochs'] = 2000
train['batch_size'] = 1
train['num_workers'] = 0
train['generator_steps'] = 2

train['lambda_rec'] = 10.0
train['lambda_gauss'] = 0.0001
train['lambda_cls'] = 0.0

train['lr_D'] = 1e-3
train['lr_D_decay_epoch'] = 500
train['lr_D_decay_rate'] = 0.998
train['beta1_D'] = 0.5
train['beta2_D'] = 0.999
train['weight_decay_D'] = 1e-5

train['lr_G'] = 5e-3
train['lr_G_decay_epoch'] = 500
train['lr_G_decay_rate'] = 0.998
train['beta1_G'] = 0.5
train['beta2_G'] = 0.999
train['weight_decay_G'] = 1e-5

# train['resume_G'] = None
# train['resume_Style_Encoder'] = None
# train['resume_D_VIS'] = None

# train['resume_optim_G'] = None
# train['resume_optim_D'] = None

train['checkpoint'] = './checkpoint_decomposer/'
train['logs'] = './logs_decomposer/'
train['out'] = '/media/hyo/文档/VIS-NIR/RESULT/out_decomposer/'

train['resume_Intrinsic_Encoder'] = 'checkpoint_decomposer/Intrinsic_Encoder_20190924.pth'
train['resume_Style_Encoder'] = 'checkpoint_decomposer/Style_Encoder_20190924.pth'
train['resume_Integrator'] = 'checkpoint_decomposer/Integrator_20190924.pth'
train['resume_D_VIS'] = 'checkpoint_decomposer/D_VIS_20190924.pth'
train['resume_D_NIR'] = 'checkpoint_decomposer/D_NIR_20190924.pth'
train['resume_D_SKETCH'] = 'checkpoint_decomposer/D_SKETCH_20190924.pth'
train['resume_D_DOMAIN'] = 'checkpoint_decomposer/D_DOMAIN_20190924.pth'
# # #
train['resume_optim_G'] = 'checkpoint_decomposer/optimizer_G_20190924.pth'
train['resume_optim_D_VIS'] = 'checkpoint_decomposer/optimizer_D_VIS_20190924.pth'
train['resume_optim_D_NIR'] = 'checkpoint_decomposer/optimizer_D_NIR_20190924.pth'
train['resume_optim_D_SKETCH'] = 'checkpoint_decomposer/optimizer_D_SKETCH_20190924.pth'
train['resume_optim_D_DOMAIN'] = 'checkpoint_decomposer/optimizer_D_DOMAIN_20190924.pth'


train['random_seed'] = None

train['cuda'] = True

# Evaluation configuration
eval = {}
eval['if_eval'] = True


