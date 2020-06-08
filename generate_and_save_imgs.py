"""This script just saves a whole bunch of images from a network."""

from torchvision import datasets, transforms, utils

import torch
from numpy import save



from dcgan_models import Generator

import argparse
import os



parser = argparse.ArgumentParser(description='Generate and save images of a network checkpoint.')
parser.add_argument('--path', metavar='DIR', default = "checkpoint.pth.tar",
                    help='path to the saved model checkpoint.')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--image-size', default=64, type=int,
                    help='Rescale images to this many pixels. (default 64)')
parser.add_argument('--noise-dim', default=100, type=int, metavar='ND',
                    help='Dimensionality of the top layer of the cortex.')
parser.add_argument('--n-filters', default=32, type=int,
                    help='Number of filters in the first conv layer of the DCGAN. Default 64')
parser.add_argument('-b', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--n-images', default=10000, type=int,
                    help='Number of images to save')
parser.add_argument('--dataset', default='cifar10',
                    choices= ['imagenet', 'folder', 'lfw', 'cifar10', 'mnist'],
                    help="What dataset are we training on?")
parser.add_argument('--noise-type', default = 'none',
                    choices= ['none', 'fixed', 'learned_by_layer', 'learned_by_channel', 'learned_filter', 'poisson'],
                    help="What variance of Gaussian noise should be applied after all layers in the "
                            "cortex? See docs for details. Default is no noise; fixed has variance 0.01")
parser.add_argument('--divisive-normalization', action='store_true',
                    help='Divisive normalization over channels, pixel by pixel. As in ProgressiveGANs')
def main(args):

    generator = load_checkpoint(args).cuda(args.gpu)

    try:
        os.mkdir("saved_images")
    except:
        pass

    # all_ims = []
    for b in range(args.n_images // args.batch_size):
        noise = torch.empty(args.b, args.noise_dim, 1, 1).normal_().cuda(args.gpu)

        imgs = generator(noise)

        # all_ims.append(imgs)
        for i in range(imgs.size(0)):
            ind = i+b*args.b
            utils.save_image(imgs[i, :, :, :], args.path + '/gen_images/{}.jpg'.format(ind),
                             normalize = True, range = (-1,1))

    # to_save = torch.stack(all_ims).numpy()
    # save('2048_images/all_imgs.png', to_save)

def load_checkpoint(args):
    """Loads a cortex from path."""
    path = args.path + '/checkpoint.pth.tar'

    generator = Generator(args.noise_dim, args.n_filters, 1 if args.dataset == 'mnist' else 3,
                          image_size=args.image_size,
                          hard_norm=args.divisive_normalization)

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        # load onto the CPU
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        raise IOError("=> no checkpoint found at '{}'".format(path))

    return generator


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


