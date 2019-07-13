# TRAIN configuration
train = {}
train['if_train'] = True
train['learning_rate'] = 1e-4
train['epochs'] = 250
train['batch_size'] = 8
train['num_workers'] = 4
train['dis_steps'] = 3

train['lambda_cyc_loss'] = 5.0
train['lambda_id_loss'] = 0
train['lambda_in_loss'] = 5.0

train['lr_D_N'] = 5e-4
train['lr_D_N_decay_epoch'] = 50
train['lr_D_N_decay_rate'] = 0.985
train['beta1_D_N'] = 0.5
train['beta2_D_N'] = 0.999
train['weight_decay_D_N'] = 1e-5

train['lr_D_V'] = 5e-4
train['lr_D_V_decay_epoch'] = 50
train['lr_D_V_decay_rate'] = 0.985
train['beta1_D_V'] = 0.5
train['beta2_D_V'] = 0.999
train['weight_decay_D_V'] = 1e-5

train['lr_G'] = 5e-4
train['lr_G_decay_epoch'] = 50
train['lr_G_decay_rate'] = 0.99
train['beta1_G'] = 0.5
train['beta2_G'] = 0.999
train['weight_decay_G'] = 1e-5

# train['lr_G_N2V'] = 1e-3
# train['lr_G_N2V_decay_epoch'] = 20
# train['beta1_G_N2V'] = 0.99
# train['beta2_G_N2V'] = 0.999
# train['weight_decay_G_N2V'] = 1e-5
#
# train['lr_G_V2N'] = 1e-3
# train['lr_G_V2N_decay_epoch'] = 20
# train['beta1_G_V2N'] = 0.99
# train['beta2_G_V2N'] = 0.999
# train['weight_decay_G_V2N'] = 1e-5

# train['resume_G_N2V'] = None
# train['resume_G_V2N'] = None
# train['resume_D_N'] = None
# train['resume_D_V'] = None
#
# train['resume_optim_G'] = None
# train['resume_optim_D_V'] = None
# train['resume_optim_D_N'] = None

train['resume_G_N2V'] = 'checkpoint/G_N2V_20190713_no_id.pth'
train['resume_G_V2N'] = 'checkpoint/G_V2N_20190713_no_id.pth'
train['resume_D_N'] = 'checkpoint/D_N_20190713_no_id.pth'
train['resume_D_V'] = 'checkpoint/D_V_20190713_no_id.pth'

train['resume_optim_G'] = 'checkpoint/optimizer_G_20190713_no_id.pth'
train['resume_optim_D_V'] = 'checkpoint/optimizer_D_V_20190713_no_id.pth'
train['resume_optim_D_N'] = 'checkpoint/optimizer_D_N_20190713_no_id.pth'

train['random_seed'] = None

train['cuda'] = True

# Evaluation configuration
eval = {}
eval['if_eval'] = True


