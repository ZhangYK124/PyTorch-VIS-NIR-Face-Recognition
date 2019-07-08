# TRAIN configuration
train = {}
train['train'] = True
train['learning_rate'] = 1e-4
train['epochs'] = 200
train['batch_size'] = 32

train['lr_D_N'] = 1e-4
train['beta1_D_N'] = 0.99
train['beta2_D_N'] = 0.999
train['weight_decay_D_N'] = 1e-5

train['lr_D_V'] = 1e-4
train['lr_G_N2V'] = 1e-4
train['lr_G_V2N'] = 1e-4

train['resume_G_N2V'] = None
train['resume_G_V2N'] = None
train['resume_D_N'] = None
train['resume_D_V'] = None

train['random_seed'] = None

train['cuda'] = True

# Evaluation configuration
eval = {}
eval['if_eval'] = True


