#!/usr/bin/env python

from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist



from dcgan_models import Inference, Generator, Discriminator
from utils import sv_img, KLfromSN, gen_surprisal, get_detached_state_dict, get_gradient_penalty, get_pixelwise_channel_norms

import argparse
import os
import random
import warnings

from orion.client import report_results
from classify_from_model import decode_classes_from_layers

parser = argparse.ArgumentParser(description='PyTorch Adversarial Wake-Sleep Training on MNIST')
parser.add_argument('-d', '--data', metavar='DIR', default = "../data",
                    help='path to dataset. Loads MNIST if nonexistant.')
parser.add_argument('--dataset', default='imagenet',
                    choices= ['imagenet', 'folder', 'lfw', 'cifar10', 'mnist'],
                    help="What dataset are we training on?")
parser.add_argument('--image-size', default=64, type=int,
                    help='Rescale images to this many pixels. (default 64)')


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr-g', default=1e-4, type=float,
                    metavar='LRC', help='initial learning rate of the generator. Default 1e-4', dest='lr_g')
parser.add_argument('--lr-d', '--learning-rate-discriminator', default=1e-4, type=float,
                    metavar='LRD', help='initial learning rate of the discriminator. Default 5e-4', dest='lr_d')
parser.add_argument('--lr-e',  default=1e-4, type=float,
                    metavar='LRD', help='initial learning rate of the encoder. Default 5e-4', dest='lr_e')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='wd')
parser.add_argument('--beta1',  default=.5, type=float,
                    help='In the adam optimizer. Default = 0.5')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run. Default 200')

parser.add_argument('--noise-dim', default=40, type=int, metavar='ND',
                    help='Dimensionality of the top layer of the cortex.')
parser.add_argument('--n-filters', default=64, type=int,
                    help='Number of filters in the first conv layer of the DCGAN. Default 64')
parser.add_argument('--surprisal-sigma', default=10, type=float,
                    metavar='ss', help='How weakly to follow the gradient of the surprisal of the lower layers '
                     'given the upper layers inference activations and the generative model.'
                     ' Specifically the variance of the implied Gaussian noise of the output (but there is no noise.'
             ' Equivalent to minimizing the reconstruction error from inference states, up one layer, and back down.')
parser.add_argument('--lamda', default=.3, type=float,
                    help='Lambda for the gradient penalty in the WGAN formulation. Only for Wasserstein loss.')


parser.add_argument('--save-imgs', action='store_true',
                    help='Save the generated images every epoch')

parser.add_argument('--soft-div-norm', default=0, type=float,
                    help='A "soft" divisive normalization over channels, pixel by pixel. A differentiable penalty.' 
                   ' If greater than zero, this is the strength by which the penalty is applied. Default 0.')

parser.add_argument('--gradient-clipping', default=50, type=float,
                    help="CLip gradients on everything at this value")


parser.add_argument('--kl-from-sn', action='store_true',
                    help='Top layer of the output is moved towards the generative prior. Batch statistics.')

# for orion
parser.add_argument('--scale-gen-surprisal-by-D', default='False',
                    help='Only update the generator when the discriminator is positive',
                    choices = ["True", "False"])
parser.add_argument('--prioritized-replay', default='False',
                    help='Replay the most surprising examples',
                    choices = ["True", "False"])
parser.add_argument('--divisive-normalization', default='False',
                    help='Divisive normalization over channels, pixel by pixel. As in ProgressiveGANs',
                    choices = ["True", "False"])
parser.add_argument('--spectral-norm', default='False',
                    help='Apply spectral norm to the inference (not disc)',
                    choices = ["True", "False"])


def train(args, inference, generator, train_loader, discriminator,
              optimizerD, optimizerG, optimizerF, epoch):
    inference.train()
    generator.train()
    discriminator.train()

    criterion = nn.L1Loss(reduction='none') if args.scale_gen_surprisal_by_D else nn.L1Loss()

    for batch, (images, _) in enumerate(train_loader):


        optimizerG.zero_grad()
        optimizerD.zero_grad()
        optimizerF.zero_grad()

        ############### WAKE #################
        # get some data
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        else:
            images = images.cuda()
        real_samples = Variable(images)

        # pass through inference
        real_out = inference(real_samples)

        Ds = discriminator(get_detached_state_dict(inference.intermediate_state_dict))
        D_loss = -(Ds).mean()
        E_loss = discriminator(inference.intermediate_state_dict).mean()

        # now update generator
        # here we have to a assume a noise model in order to calculate p(h_1 | h_2 ; G)
        # with Gaussian we have log p  = MSE between actual and predicted
        surp = None if not args.scale_gen_surprisal_by_D else Ds
        G_loss = gen_surprisal(inference.intermediate_state_dict, generator, args.surprisal_sigma, criterion,
                               surp, detach_inference=True, )

        if args.kl_from_sn:
            E_loss = E_loss + KLfromSN(real_out)

        ############### SLEEP ##################

        # fantasize

        noise = torch.empty(real_samples.size(0), args.noise_dim, 1, 1).normal_().to(real_samples.device)

        # prioritized replay
        if args.prioritized_replay:
            most_surprising_idx = torch.sort(Ds, dim=0)[1][:real_samples.size(0) // 2].squeeze()
            noise[:real_samples.size(0) // 2] = real_out[most_surprising_idx].detach()

        generated_down = generator(noise)

        Ds_gen = discriminator(get_detached_state_dict(generator.intermediate_state_dict))
        D_loss += (Ds_gen).mean()

        # center the discriminator
        if args.scale_gen_surprisal_by_D:
            D_loss += torch.norm((Ds_gen).mean() + (Ds).mean())

        D_loss += get_gradient_penalty(discriminator, generator.intermediate_state_dict, lamda = args.lamda, p=1)
        D_loss += get_gradient_penalty(discriminator, inference.intermediate_state_dict, lamda = args.lamda, p=1)

        D_loss.backward()
        if args.gradient_clipping > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(),
                                     args.gradient_clipping, "inf")
        optimizerD.step()

        E_loss.backward(retain_graph=True)
        if args.gradient_clipping > 0:
            nn.utils.clip_grad_norm_(inference.parameters(),
                                     args.gradient_clipping, "inf")
        optimizerF.step()

        optimizerG.zero_grad()
        G_loss.backward()
        optimizerG.step()


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.scale_gen_surprisal_by_D = args.scale_gen_surprisal_by_D == "True"
    args.prioritized_replay = args.prioritized_replay == "True"
    args.divisive_normalization = args.divisive_normalization == "True"
    args.spectral_norm = args.spectral_norm == "True"


    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    # ----- Get dataset ------ #

    image_size = args.image_size

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if args.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]))
        nc = 3
    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc = 3

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root=args.data, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
        nc = 1

    assert train_dataset

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # ----- Create models ------ #


    inference = Inference(args.noise_dim, args.n_filters, nc, image_size=image_size,
                          noise_before=False, hard_norm=args.divisive_normalization,
                                            spec_norm = args.spectral_norm)
    generator = Generator(args.noise_dim, args.n_filters, nc, image_size=image_size,
                          hard_norm=args.divisive_normalization)

    discriminator = Discriminator(args.noise_dim, args.n_filters, nc, image_size=image_size,
                                  hard_norm=args.divisive_normalization, hidden_dim=128)



    # get to proper GPU
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            inference.cuda(args.gpu)
            generator.cuda(args.gpu)

            discriminator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            inference = torch.nn.parallel.DistributedDataParallel(inference, device_ids=[args.gpu])
            generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu])
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
        else:
            inference.cuda()
            generator.cuda()

            discriminator.cuda()

            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            generator = torch.nn.parallel.DistributedDataParallel(generator)
            inference = torch.nn.parallel.DistributedDataParallel(inference)

            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        inference = inference.cuda(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
        generator = generator.cuda(args.gpu)

    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        inference = torch.nn.DataParallel(inference).cuda()
        generator = torch.nn.DataParallel(generator).cuda()
        discriminator = torch.nn.DataParallel(discriminator).cuda()


    # ------ Build optimizer ------ #
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999), weight_decay = args.wd)
    # we want the lr to be slower for upper layers as they get more gradient flow
    optimizerG = optim.Adam([{'params':mod.parameters(),'lr': args.lr_g * (5-i)}
                                    for i,mod in enumerate(generator.listed_modules)],
                     lr=args.lr_g, betas=(args.beta1, 0.999), weight_decay = args.wd)

    # similarly for the encoder lower layers should have have slower lrs
    optimizerF = optim.Adam([{'params':mod.parameters(),'lr': args.lr_e * (i+1)}
                                    for i,mod in enumerate(inference.listed_modules)],
                     lr=args.lr_e, betas=(args.beta1, 0.999), weight_decay = args.wd)

    # ------ optionally resume from a checkpoint ------- #
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']

            inference.load_state_dict(checkpoint['inference_state_dict'])
            generator.load_state_dict(checkpoint['generator_state_dict'])

            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD'])
            optimizerG.load_state_dict(checkpoint['optimizerG'])
            optimizerF.load_state_dict(checkpoint['optimizerF'])
            train_history = checkpoint['train_history']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        train_history = {'D_losses': [],
                         'GF_losses': [],
                         'ML_losses': [],
                         'reconstruction_error': []}
        args.start_epoch = 0

    decoding_error_history = []

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(args, inference, generator, train_loader, discriminator,
                        optimizerD, optimizerG, optimizerF, epoch)
        generator.eval()
        inference.eval()

        if args.save_imgs:
            try:
                os.mkdir("gen_images")
            except:
                pass
            noise = torch.empty(100,args.noise_dim,1,1).normal_().cuda()
            to_visualize = generator(noise).detach().cpu()
            grid = utils.make_grid(to_visualize,
                                  nrow=10, padding=5, normalize=True,
                                  range=None, scale_each=False, pad_value=0)
            sv_img(grid, "gen_images/imgs_epoch{}.png".format(epoch), epoch)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if (epoch % 5 == 0) or (epoch == args.epochs-1):
                # how well can we decode from layers?
                accuracies = decode_classes_from_layers(  args.gpu,
                                                              inference,
                                                              image_size,
                                                              args.n_filters,
                                                              args.noise_dim,
                                                              args.data,
                                                                args.dataset,
                                                              nonlinear = False,
                                                              lr = 1,
                                                              folds = 8,
                                                              epochs = 20,
                                                              hidden_size = 1000,
                                                              wd = 1e-3,
                                                              opt = 'sgd',
                                                              lr_schedule = True,
                                                              verbose=False,
                                                              batch_size=args.batch_size,
                                                              workers=args.workers)
                print("Epoch {}".format(epoch))
                for i in range(6):
                    print("Layer{}: Accuracy {} +/- {}".format(i, accuracies.mean(dim=0)[i],accuracies.std(dim=0)[i]))
                decoding_error_history.append(accuracies.mean(dim=0).detach().cpu())

            torch.save({
                'epoch': epoch + 1,
                'inference_state_dict': inference.state_dict(),
                'generator_state_dict': inference.state_dict(),

                'discriminator_state_dict': discriminator.state_dict(),
                'args': args,
                'optimizerD': optimizerD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerF': optimizerF.state_dict(),
                'train_history': {"decoding_error_history": decoding_error_history}
            }, 'checkpoint.pth.tar')

    # For orion. Don't include lowest layer
    best_accuracy = torch.max(accuracies.mean(dim=0)[1:]).item()
    error_rate = 100 - best_accuracy
    report_results([dict(
        name='test_error_rate',
        type='objective',
        value=error_rate)])

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

