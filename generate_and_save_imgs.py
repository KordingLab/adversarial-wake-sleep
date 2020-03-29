"""This script just saves a whole bunch of images from a network."""

from torchvision import datasets, transforms, utils

import torch
from numpy import save



from dcgan_models import DeterministicHelmholtz

import argparse
import os



parser = argparse.ArgumentParser(description='Generate and save images of a network checkpoint.')
parser.add_argument('--path', metavar='DIR', default = "checkpoint.pth.tar",
                    help='path to the saved model checkpoint.')
parser.add_argument('--image-size', default=64, type=int,
                    help='Rescale images to this many pixels. (default 64)')
parser.add_argument('--noise-dim', default=40, type=int, metavar='ND',
                    help='Dimensionality of the top layer of the cortex.')
parser.add_argument('--n-filters', default=64, type=int,
                    help='Number of filters in the first conv layer of the DCGAN. Default 64')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')

parser.add_argument('--noise-type', default = 'none',
                    choices= ['none', 'fixed', 'learned_by_layer', 'learned_by_channel', 'learned_filter', 'poisson'],
                    help="What variance of Gaussian noise should be applied after all layers in the "
                            "cortex? See docs for details. Default is no noise; fixed has variance 0.01")

def main(args):
    image_size = args.image_size
    noise_dim = args.noise_dim
    n_filters = args.n_filters
    cortex = DeterministicHelmholtz(noise_dim, n_filters, 3, noise_type=args.noise_type,
                                    image_size=image_size).eval()

    _ = load_checkpoint(args.path, cortex)

    try:
        os.mkdir("saved_images")
    except:
        pass

    # all_ims = []
    for b in range(4096 // args.batch_size):
        noise = torch.empty(args.batch_size,args.noise_dim,1,1).normal_()

        imgs = cortex.generate(noise).detach().cpu()
        # all_ims.append(imgs)
        for i in range(imgs.size(0)):
            ind = i+b*args.batch_size
            utils.save_image(imgs[i, :, :, :], 'saved_images/{}.png'.format(ind))

    # to_save = torch.stack(all_ims).numpy()
    # save('2048_images/all_imgs.png', to_save)

def load_checkpoint(path, cortex):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        a = torch.ones(1).cuda()

        checkpoint = torch.load(path, a.device)

        cortex.load_state_dict(checkpoint['cortex_state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
        train_history = checkpoint['train_history']
        return train_history
    else:
        print("=> no checkpoint found at '{}'".format(path))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


