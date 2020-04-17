# LIA

Coding implemention and datasets for ["LIA: Latently Invertible Autoencoder with Adversarial Learning"](https://arxiv.org/abs/1906.08090), which is based on DCGAN framework and Anmie Faces Datasets.


## Requirements

Software:
```
Python3
Pytorch >= 1.0
argparse == 1.1
```

## Prepare

* Download the anmie faces data from [here](https://pan.baidu.com/s/1OOi5GVkGWbNW7HrpY0h2DQ) , password : **bvqu**
* Unzip the zipfile ``faces``   directory
* Save the unzipped files ``faces`` to the ``\data`` directory

## training the model

###  example

```sh
CUDA_VISIBLE_DEVICES='0'  python3 -u train.py \
    --nz 128 \
    --ngf 64 \
    --ndf 64 \
    --epoch 30 \
    --outf 'outputs/'\
    > ./outputs/LIA_running.log &
```

Options of ``train.py``:
```
useage: [--batchSize] - training batch size, default = 64.
        [--imageSize] - the transformed image size, default = 128.
        [--nz] - the size of the latent z vector, default = 128.
        [--ngf] - the size of f and g function's convolution process channel unit, default = 64.
        [--ngf] - the size of c function's convolution process channel unit, default = 64.
        [--epoch] - the number of training epoches, default = 30.
        [--lr] - the learning rate, default = 0.0002.
        [--beta1] - beta1 for adam. default=0.5.
        [--data_path] - the folder to train data.
        [--outf] - the folder to output images and model checkpoints.
        [--gamma] - the weight of regularization term. default=3.
        [--alpha] - the weight of feature extraction term. default=0.001.
```








