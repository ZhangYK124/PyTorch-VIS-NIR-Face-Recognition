# TRAIN configuration
train = {}
train['if_train'] = True
train['learning_rate'] = 1e-4
train['epochs'] = 200
train['batch_size'] = 6
train['num_workers'] = 4

train['lambda_cyc_loss'] = 10.0
train['lambda_id_loss'] = 5.0

train['lr_D_N'] = 1e-3
train['lr_D_N_decay_epoch'] = 100
train['beta1_D_N'] = 0.99
train['beta2_D_N'] = 0.999
train['weight_decay_D_N'] = 1e-5

train['lr_D_V'] = 1e-3
train['lr_D_V_decay_epoch'] = 100
train['beta1_D_V'] = 0.99
train['beta2_D_V'] = 0.999
train['weight_decay_D_V'] = 1e-5

train['lr_G'] = 1e-3
train['lr_G_decay_epoch'] = 100
train['beta1_G'] = 0.99
train['beta2_G'] = 0.999
train['weight_decay_G'] = 1e-5

train['lr_G_N2V'] = 1e-3
train['lr_G_N2V_decay_epoch'] = 100
train['beta1_G_N2V'] = 0.99
train['beta2_G_N2V'] = 0.999
train['weight_decay_G_N2V'] = 1e-5

train['lr_G_V2N'] = 1e-3
train['lr_G_V2N_decay_epoch'] = 100
train['beta1_G_V2N'] = 0.99
train['beta2_G_V2N'] = 0.999
train['weight_decay_G_V2N'] = 1e-5

train['resume_G_N2V'] = None
train['resume_G_V2N'] = None
train['resume_D_N'] = None
train['resume_D_V'] = None

train['random_seed'] = None

train['cuda'] = True

# Evaluation configuration
eval = {}
eval['if_eval'] = True


