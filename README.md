
# Adversarial Wake-Sleep

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

This repo requires pytorch v1.3 or greater. To install requirements in a new Conda environment run

```setup
conda env create --file requirements.yaml
```

## Training

### Adversarial Wake-Sleep

To train a DCGAN-style architecture with pure Adversarial Wake-Sleep to generate MNIST digits as in the paper, run this command:

```train
python train.py --dataset mnist --data [PATH-TO-MNIST] --image-size 32 --epochs 30 --n-filters 32 --noise-dim 40 --lr-g 0.00001 --lr-d 0.0001 --lr-e 0.0003 --beta1 0.64  --beta2 .999 --lamda .9 --kl-from-sn --divisive-normalization True --save-imgs
```

This is the hyperparameter setting used in the paper.

### Doubly Adversarial Wake-Sleep

We can force the inference network to doubly act as a discriminator on images by setting `gamma` less than 1. `gamma=0` is a standard WGAN-GP.
Here on MNIST
```train
python train.py --dataset mnist --data [PATH-TO-MNIST] -b 128 --image-size 32 --epochs 100 --n-filters 32 --noise-dim 40 --lr-g 0.0002 --lr-d 0.00001 --lr-e 0.0008 --lr-rd 0.00001 --beta1 0  --beta2 .9 --lamda .7 --lamda2 7 --kl-from-sn --divisive-normalization True --save-imgs --gamma 0.9```
```
or CIFAR-10
```train
python train.py --dataset cifar10 --data [PATH-TO-CIFAR10] -b 128 --image-size 32 --epochs 200 --n-filters 128 --noise-dim 100 --lr-g 0.0002 --lr-d 0.00001 --lr-e 0.0008 --lr-rd 0.00001 --beta1 0  --beta2 .9 --lamda .7 --lamda2 7 --kl-from-sn --divisive-normalization True --save-imgs --gamma 0.9```
```
## Evaluation:

To save a batch of 10,000 generated images (for evaluating the FID score using code from e.g. [TTUR](https://github.com/bioinf-jku/TTUR)), run this command:

```eval
python generate_and_save_imgs.py --dataset cifar10 --path [PATH-TO-MODEL-CHECKPOINT] --image-size 32 --n-filters 128 --noise-dim 100 --divisive-normalization
```
This loads a checkpoint with the specified architecture. Make sure to ensure this architecture is the same.

To get the linear separability of classes layers, call this command to load a model and run a linear SVM (requires scikit-learn):
```
python SVM_classify_from_model.py --dataset cifar10 --path [PATH-TO-MODEL-CHECKPOINT] --data [PATH-TO-CIFAR] -b 512 --image-size 32 --n-filters 128 --noise-dim 100 --divisive-normalization --n-folds 10 --alpha 0.0001 --epochs 5
```
