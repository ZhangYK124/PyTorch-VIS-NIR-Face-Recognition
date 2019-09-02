# VIS-NIR Face Recognition, in PyTorch

## Update Logs

### 201907151736

#### Output

    - out: no final residual block
    - out1: final resudial block * 3
#### Logs
    - batch_size = 1,which is same as cyclegan.
    - instance norm
    - no dropout
    - discriminator is utilized to classify real/fake vis/nir and real/fake vis/nir left/right
    eyes rather than real/fake local vis/nir(concatenation of left and right eye).

### 201907161702

#### Outputs
    - Out2
    - Logs2

#### Logs
    
    - increase weight of generative loss function for the whole face (*3)
    
    
### 201907230900

#### Logs

    - train local path and global path saparately
    