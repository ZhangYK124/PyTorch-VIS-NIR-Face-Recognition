# VIS-NIR Face Recognition, in PyTorch

## Update Logs

### 201907151736
    - batch_size = 1,which is same as cyclegan.
    - instance norm
    - no dropout
    - discriminator is utilized to classify real/fake vis/nir and real/fake vis/nir left/right
    eyes rather than real/fake local vis/nir(concatenation of left and right eye).
