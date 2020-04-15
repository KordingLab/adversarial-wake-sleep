#!/usr/bin/env python

from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist



from dcgan_models import DeterministicHelmholtz, Discriminator
from utils import sv_img, get_gradient_penalty

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
parser.add_argument('--lr-d', '--learning-rate-discriminator', default=5e-4, type=float,
                    metavar='LRD', help='initial learning rate of the discriminator. Default 5e-4', dest='lr_d')
parser.add_argument('--lr-e',  default=5e-4, type=float,
                    metavar='LRD', help='initial learning rate of the encoder. Default 5e-4', dest='lr_e')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='wd')
parser.add_argument('--beta1',  default=.5, type=float,
                    help='In the adam optimizer. Default = 0.5')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run. Default 200')
parser.add_argument('--loss-type', default='BCE',
                    choices= ['BCE', 'wasserstein', 'hinge'],
                    help="The form of the minimax loss function. BCE = binary cross entropy of the original GAN")
parser.add_argument('--detailed-logging', action='store_true',
                    help='Whether to store detailed information about training metrics inside of the cortex object')
parser.add_argument('--noise-dim', default=40, type=int, metavar='ND',
                    help='Dimensionality of the top layer of the cortex.')
parser.add_argument('--n-filters', default=64, type=int,
                    help='Number of filters in the first conv layer of the DCGAN. Default 64')
parser.add_argument('--surprisal-sigma', default=10, type=float,
                    metavar='ss', help='How weakly to follow the gradient of the surprisal of the lower layers '
                     'given the upper layers inference activations and the generative model.'
                     ' Specifically the variance of the implied Gaussian noise of the output (but there is no noise.'
             ' Equivalent to minimizing the reconstruction error from inference states, up one layer, and back down.')
parser.add_argument('--minimize-generator-surprisal', action='store_true',
                    help='Minimize generator surprisal using value of sigma set.')
parser.add_argument('--minimize-inference-surprisal', action='store_true',
                    help='Minimize generator surprisal using value of sigma set.')
parser.add_argument('--lamda', default=.1, type=float,
                    help='Lambda for the gradient penalty in the WGAN formulation. Only for Wasserstein loss.')

parser.add_argument('--noise-type', default = 'none',
                    choices= ['none', 'fixed', 'learned_by_layer', 'learned_by_channel', 'learned_filter', 'poisson'],
                    help="What variance of Gaussian noise should be applied after all layers in the "
                            "cortex? See docs for details. Default is no noise; fixed has variance 0.01")


parser.add_argument('--no-backprop-through-full-cortex', action='store_true',
                    help='Only backprop through the local (layerwise) discriminator to parameters in that same '
                     'layer of the cortex. For biological plausibility.')
parser.add_argument('--only-backprop-generator', action='store_true',
                    help='Only backprop through the local (layerwise) discriminator to parameters in that same '
                     'layer of the cortex for the inference. For the generator we backprop all the way to the top.')

parser.add_argument('--quiet', action='store_true',
                    help='Do not print the surprisal stats.')
parser.add_argument('--save-imgs', action='store_true',
                    help='Save the generated images every epoch')
parser.add_argument('--label-smoothing', action='store_true',
                    help='Make the discriminator be less confident by randomly switching labels with p=.1')
parser.add_argument('--minibatch-std-dev', action='store_true',
                    help='Evaluate the standard deviation of the features in the 2nd-to-last layer of the discriminator'
                    ' and add it as a feature. As in progressiveGAN.')
parser.add_argument('--divisive-normalization', action='store_true',
                    help='Divisive normalization over channels, pixel by pixel. As in ProgressiveGANs')
parser.add_argument('--soft-div-norm', default=0, type=float,
                    help='A "soft" divisive normalization over channels, pixel by pixel. A differentiable penalty.' 
                   ' If greater than zero, this is the strength by which the penalty is applied. Default 0.')
parser.add_argument('--gradient-clipping', default=0, type=float,
                    help="CLip gradients on everything at this value")
parser.add_argument('--amsgrad', action='store_true',
                    help="Use AMSgrad?")
parser.add_argument('--update-ratio', default=1, type=float,
                    help="Ratio of discriminator updates to the others.")

parser.add_argument('--kl-from-sn', action='store_true',
                    help="Instead of an actual (linear) discriminator, calculate the true KL divergence from "
                         "the prior p(z) analytically.")
parser.add_argument('--spectral-norm', action='store_true',
                    help="Spectral norm on E and D")

parser.add_argument('--only-latents', action='store_true',
                    help="When activated, it's only the inputs and top layer we care about. No middle. Should "
                         "be the same as previously published algorithms; i.e. should work lol")
parser.add_argument('--reparam-trick', action='store_true',
                    help="When activated, uses the VAE loss function on the encoder's outputs. "
                         "Interprets half the channels as variance of noise to-be-added, etc.")



def train(args, cortex, train_loader, discriminator,
              optimizerD, optimizerG, optimizerE, epoch):

    noise_layer = len(cortex.generator.listed_modules)

    cortex.train()
    discriminator.train()
    print("Reconstructions, E discriminates too, and KL/wass of wake from prior")

    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    for batch, (images, _) in enumerate(train_loader):
        cortex.log_weight_alignment()
        batch_size = images.size()[0]

        optimizerG.zero_grad()
        optimizerE.zero_grad()
        optimizerD.zero_grad()

        ############### WAKE #################
        # get some data
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        else:
            images = images.cuda()
        real_samples = Variable(images)

        if args.label_smoothing:
            #with prob .1 we switch the labels
            p = torch.ones(batch_size,1) * .9
            alpha = torch.bernoulli(p) * 2 - 1
        else:
            alpha = torch.ones(batch_size,1)
        alpha = alpha.to(real_samples.device)

        # pass through inference
        inferred_z = cortex.inference(real_samples)

        # now update generator to reduce its surprisal given the inferred state
        # here we have to a assume a noise model in order to calculate p(h_1 | h_2 ; G)
        # with Gaussian we have log p  = MSE between actual and predicted
        ML_loss = torch.zeros(1)
        if args.minimize_generator_surprisal or args.detailed_logging or args.soft_div_norm>0:
            ML_loss = cortex.generator_surprisal()
        if args.minimize_generator_surprisal:
            ML_loss.backward()
            optimizerG.step()
            optimizerG.zero_grad()

        ## argmax_D_E D(E(real_samples))
        # recall the discriminator outputs (ideally) >0 for through encoder and <0 for generative posterior
        # so here E wants to minimize
        # Also when D is the straight divergence of the prior (i.e. args.kl_from_sn ==True)
        # E wants to minimize

        if batch % args.update_ratio == 0:
            wake_E_loss = (alpha * discriminator(inferred_z)).mean()
            wake_E_loss.backward(retain_graph = True)
            optimizerD.zero_grad()

            if args.reparam_trick:
                reconstruction_loss = 10*l1(cortex.generator(add_noise(inferred_z)), real_samples)
            else:
                reconstruction_loss = 10*l1(cortex.generator(inferred_z), real_samples)
            reconstruction_loss.backward(retain_graph = True)
            if args.reparam_trick:
                optimizerG.step()
            optimizerG.zero_grad()
            if batch % 100 == 0:
                print("WAKE. Reconst. {:.2f}. closeness to prior {:.2f}".format(reconstruction_loss.item(),
                                                                                wake_E_loss.item()))
            optimizerE.step()
            optimizerE.zero_grad()


        if not args.kl_from_sn:
            # stabilize only D with GP
            gp = get_gradient_penalty(discriminator, cortex, args.lamda, 'inference', only_output=True)
            wake_D_loss =  - (alpha * discriminator(inferred_z.detach())).mean() + gp
            wake_D_loss.backward()
            optimizerD.step()
            optimizerD.zero_grad()
            optimizerE.zero_grad()

        ############### SLEEP #################

        # fantasize
        noise = torch.empty(real_samples.size(0),args.noise_dim,1,1).normal_().to(real_samples.device)
        fake_inputs = cortex.generator(noise)

        # learn to divisive normalize both feedforward and feedback?
        if args.soft_div_norm > 0:
            div_norm_loss = cortex.get_pixelwise_channel_norms() * args.soft_div_norm
            div_norm_loss.backward(retain_graph = True)

        if args.minimize_inference_surprisal:
            ML_loss = cortex.inference_surprisal()
            ML_loss.backward()
        # if not args.quiet:
        #     if batch % 100 == 0:
        #         print("Epoch {} Batch {} Overall surprisal {:.2f}".format(epoch, batch, ML_loss.item()))



        # train discriminator to recognize noise as noise
        if not args.kl_from_sn:
            sleep_D_loss = (alpha * discriminator(noise)).mean()
            sleep_D_loss += get_gradient_penalty(discriminator, cortex, args.lamda, 'generation', only_output=True)

            sleep_D_loss.backward()
            optimizerD.step()
            optimizerD.zero_grad()


        ## argmax_G argmin_D,E  D(E(fake_samples))
        # recall the discriminator outputs (ideally) >0 for through encoder and <0 for generative posterior
        # so here G wants to minimize D(E(G(z))) but E maximize
        # Also when D is the straight divergence of the prior (i.e. args.kl_from_sn ==True)
        # G wants to minimize while E wants to maximize
        if batch % args.update_ratio == 0:
            if not args.only_latents:
                inferred_zs = cortex.pass_state_back_up()
                inferred_z_fake = inferred_zs[0]

                D = 0
                for z in inferred_zs:
                    D = D + discriminator(z)
                D = alpha * D

            else:
                inferred_z_fake = cortex.inference(fake_inputs)

                D = alpha * discriminator(inferred_z_fake)

            if args.reparam_trick:
                reconstruction_loss = torch.zeros(1).cuda(args.gpu)
            else:
                reconstruction_loss = 10 * mse(inferred_z_fake, noise)

            # E, D works to minimize, but G works to maximize, the output of D
            sleep_D_loss_recon = 0.1 * D.mean()

            # First the generator
            G_loss = reconstruction_loss + sleep_D_loss_recon
            G_loss.backward(retain_graph = True)
            optimizerG.step()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            optimizerE.zero_grad()


            # then the discriminator
            E_D_loss = - sleep_D_loss_recon
            E_D_loss.backward(retain_graph = True)
            optimizerE.step()
            if not args.kl_from_sn:
                gp = get_gradient_penalty(discriminator, cortex, args.lamda, 'inference',
                                                     only_output=True)

                gp.backward()
                optimizerD.step()
                optimizerD.zero_grad()

            if batch % 100 == 0:
                print("SLEEP.  reconst. {:.2f} closeness to prior of recont. {:.2f}".format(reconstruction_loss.item()
                                                                                                ,sleep_D_loss_recon.item()))


def add_noise(x):
    "Assumes the second half of x is variances"
    n_channels = x.size()[1]
    assert n_channels % 2 == 0

    mu = x[:, :n_channels // 2, :, :]
    sigma = x[:, n_channels // 2:, :, :]

    noise = torch.empty_like(mu).normal_()
    out = mu + noise * sigma
    return out

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

    if args.only_latents:
        assert not args.minimize_inference_surprisal and not args.minimize_generator_surprisal

    # Remember that to use multiprocessing, the module we're calling it on must only be ever be called via forward.'
    # No calling its internal methods

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

    # False if no bp thru cortex, or if only bp gen, else true
    bp_thru_inf = (not args.no_backprop_through_full_cortex) and (not args.only_backprop_generator)
    bp_thru_gen = (not args.no_backprop_through_full_cortex)

    bn = False if args.loss_type == 'wasserstein' else True


    cortex = DeterministicHelmholtz(args.noise_dim, args.n_filters, nc,
                                    image_size=image_size,
                                    surprisal_sigma=args.surprisal_sigma,
                                    detailed_logging=args.detailed_logging,
                                    backprop_to_start_inf=bp_thru_inf,
                                    backprop_to_start_gen=bp_thru_gen,
                                    noise_type = args.noise_type,
                                    batchnorm = bn,
                                    normalize = args.divisive_normalization,
                                    spectral_norm = args.spectral_norm,
                                    reparam_trick = args.reparam_trick)

    discriminator = Discriminator(args.noise_dim,
                                 eval_std_dev=args.minibatch_std_dev,
                                 spectral_norm = args.spectral_norm,
                                 detailed_logging = args.detailed_logging or (args.soft_div_norm>0),
                                 KL_from_sn = args.kl_from_sn,
                                 reparam_trick = args.reparam_trick)

    # get to proper GPU
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            cortex.cuda(args.gpu)
            discriminator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            cortex = torch.nn.parallel.DistributedDataParallel(cortex, device_ids=[args.gpu])
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
        else:
            cortex.cuda()
            discriminator.cuda()

            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            cortex = torch.nn.parallel.DistributedDataParallel(cortex)
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cortex = cortex.cuda(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        cortex = torch.nn.DataParallel(cortex).cuda()
        discriminator = torch.nn.DataParallel(discriminator).cuda()


    # ------ Build optimizer ------ #
    if isinstance(cortex, torch.nn.parallel.DistributedDataParallel):
        generator_params = cortex.module.generator.parameters()
        inference_params = cortex.module.inference.parameters()
        discriminator_params = discriminator.parameters()
    else:
        generator_params = cortex.generator.parameters()
        inference_params = cortex.inference.parameters()
        discriminator_params = discriminator.parameters()

    optimizerD = optim.Adam(discriminator_params, lr=args.lr_d, betas=(args.beta1, 0.999), weight_decay = args.wd,
                                                                                           amsgrad = args.amsgrad)
    optimizerG = optim.Adam(generator_params,     lr=args.lr_g, betas=(args.beta1, 0.999), weight_decay = args.wd,
                                                                                            amsgrad = args.amsgrad)
    optimizerF = optim.Adam(inference_params,     lr=args.lr_e, betas=(args.beta1, 0.999), weight_decay = args.wd,
                                                                                            amsgrad = args.amsgrad)

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

            cortex.load_state_dict(checkpoint['cortex_state_dict'])
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

    t2 = 2 if args.reparam_trick else 1


    if args.detailed_logging:
        # how well can we decode from layers?
        accuracies = decode_classes_from_layers(args.gpu,
                                                cortex,
                                                image_size,
                                                args.n_filters,
                                                args.noise_dim *t2,
                                                args.data,
                                                args.dataset,
                                                nonlinear=False,
                                                lr=1,
                                                folds=3,
                                                epochs=20,
                                                hidden_size=1000,
                                                wd=1e-3,
                                                opt='sgd',
                                                lr_schedule=True,
                                                verbose=False,
                                                batch_size=args.batch_size,
                                                workers=args.workers)
        for i in range(6):
            print("Layer{}: Accuracy {} +/- {}".format(i, accuracies.mean(dim=0)[i], accuracies.std(dim=0)[i]))
        decoding_error_history.append(accuracies.mean(dim=0).detach().cpu())

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(args, cortex, train_loader, discriminator,
              optimizerD, optimizerG, optimizerF, epoch)

        if args.save_imgs:
            try:
                os.mkdir("gen_images")
            except:
                pass
            noise = torch.empty(100, args.noise_dim, 1, 1).normal_().cuda(args.gpu)
            to_visualize = cortex.generator(noise).detach().cpu()
            grid = utils.make_grid(to_visualize,
                                  nrow=10, padding=5, normalize=True,
                                  range=None, scale_each=False, pad_value=0)
            sv_img(grid, "gen_images/imgs_epoch{}.png".format(epoch), epoch)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if args.detailed_logging or (epoch == args.epochs-1):
                # how well can we decode from layers?
                accuracies = decode_classes_from_layers(  args.gpu,
                                                              cortex,
                                                              image_size,
                                                              args.n_filters,
                                                              args.noise_dim * t2,
                                                              args.data,
                                                              args.dataset,
                                                              nonlinear = False,
                                                              lr = 1,
                                                              folds = 3,
                                                              epochs = 20,
                                                              hidden_size = 1000,
                                                              wd = 1e-3,
                                                              opt = 'sgd',
                                                              lr_schedule = True,
                                                              verbose=False,
                                                              batch_size=args.batch_size,
                                                              workers=args.workers)
                for i in range(6):
                    print("Layer{}: Accuracy {} +/- {}".format(i, accuracies.mean(dim=0)[i],accuracies.std(dim=0)[i]))
                decoding_error_history.append(accuracies.mean(dim=0).detach().cpu())

            torch.save({
                'epoch': epoch + 1,
                'cortex_state_dict': cortex.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'args': args,
                'optimizerD': optimizerD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerF': optimizerF.state_dict(),
                'train_history': {"disc_loss" : discriminator.intermediate_Ds,
                                  "surprisals": cortex.intermediate_surprisals,
                                  "reconstructions": cortex.intermediate_reconstructions,
                                  "weight_alignment": cortex.weight_alignment,
                                  "cortex_channel_magnitudes": cortex.channel_norms,
                                  "decoding_error_history": decoding_error_history}
            }, 'checkpoint.pth.tar')

    # For orion. Don't include lowest layer
    best_accuracy = accuracies.mean(dim=0)[-1].item()
    error_rate = 100 - best_accuracy
    report_results([dict(
        name='test_error_rate',
        type='objective',
        value=error_rate)])

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

