from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist



from dcgan_models import DeterministicHelmholtz, Discriminator, HistoricalAverageDiscriminatorLoss
from utils import sv_img

import argparse
import os
import random
import warnings


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
parser.add_argument('--lr-c', '--learning-rate-cortex', default=1e-4, type=float,
                    metavar='LRC', help='initial learning rate of the inference and generator. Default 1e-4', dest='lr_c')
parser.add_argument('--lr-d', '--learning-rate-discriminator', default=5e-4, type=float,
                    metavar='LRD', help='initial learning rate of the discriminator. Default 5e-4', dest='lr_d')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='wd')
parser.add_argument('--beta1',  default=.5, type=float,
                    help='In the adam optimizer. Default = 0.5')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run. Default 90')
parser.add_argument('--loss-type', default='BCE',
                    choices= ['BCE', 'wasserstein', 'hinge'],
                    help="The form of the minimax loss function. BCE = binary cross entropy of the original GAN")
parser.add_argument('--detailed-logging', action='store_true',
                    help='Whether to store detailed information about training metrics inside of the cortex object')
parser.add_argument('--noise-dim', default=40, type=int, metavar='ND',
                    help='Dimensionality of the top layer of the cortex.')
parser.add_argument('--disc-hidden-dim', default=32, type=int,
                    help='Dimensionality oftthe first conv layer in *each* discriminator. Default 32')
parser.add_argument('--n-filters', default=64, type=int,
                    help='Number of filters in the first conv layer of the DCGAN. Default 64')
parser.add_argument('--surprisal-sigma', default=1, type=float,
                    metavar='ss', help='How weakly to follow the gradient of the surprisal of the lower layers '
                     'given the upper layers inference activations and the generative model.'
                     ' Specifically the variance of the implied Gaussian noise of the output (but there is no noise.'
             ' Equivalent to minimizing the reconstruction error from inference states, up one layer, and back down.')
parser.add_argument('--minimize-generator-surprisal', action='store_true',
                    help='Minimize generator surprisal using value of sigma set.')
parser.add_argument('--lamda', default=.1, type=float,
                    help='Lambda for the gradient penalty in the WGAN formulation. Only for Wasserstein loss.')
parser.add_argument('--noise-sigma', default=0, type=float,
                    help='If set, add Gaussian noise with this variance to the pre-Relu activations of both passes.')


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
parser.add_argument('--historical-averaging',  default=0, type=float,
                    help='Make the discriminator move a little more slowly. Used if value > 0')

parser.add_argument('--sequential-training', action='store_true',
                    help='Train the first layer to completion, then add on the other ones. '
                    'Noise in injected at the top of the layer being trained (as the prior). Other layers are frozen.')

# parser.add_argument('--only-match-F-to-G', action='store_true',
#                     help='Instead of mutually matching the network states to each other, only optimize'
#                     ' F towards G. G is trained just as a GAN on the inputs, via backprop.')

def train(args, cortex, train_loader, discriminator,
              optimizerD, optimizerG, optimizerF, epoch, ml_after_epoch = -1):

    noise_layer = len(cortex.generator.listed_modules)

    cortex.train()
    discriminator.train()

    if args.historical_averaging > 0:
        ha_loss = HistoricalAverageDiscriminatorLoss(args.historical_averaging)
        if args.gpu is not None:
            ha_loss = ha_loss.cuda(args.gpu)
        else:
            ha_loss = ha_loss.cuda()


    for batch, (images, _) in enumerate(train_loader):
        cortex.log_weight_alignment()

        batch_size = images.size()[0]

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
        inferred_noise = cortex.inference(real_samples)

        cortex.log_layerwise_reconstructions()

        # now update generator with the wake-sleep step
        # here we have to a assume a noise model in order to calculate p(h_1 | h_2 ; G)
        # with Gaussian we have log p  = MSE between actual and predicted
        ML_loss = cortex.generator_surprisal()
        if epoch > ml_after_epoch:
            ML_loss.backward()
            optimizerG.step()
            optimizerG.zero_grad()

        # We could update the discriminator here too, if we want.
        # For efficiency I'm putting it later (in the 'sleep') section

        ############### SLEEP ##################

        # fantasize

        generated_input = cortex.noise_and_generate(noise_layer)

        if args.label_smoothing:
            #with prob .1 we switch the labels
            p = torch.ones(batch_size,1) * .9
            alpha = torch.bernoulli(p) * 2 - 1
        else:
            alpha = torch.ones(batch_size,1)
        alpha = alpha.to(real_samples.device)

        # check out what the discriminator says
        if args.loss_type == 'hinge':
            disc_loss = nn.ReLU()(1.0 - alpha * discriminator(cortex.inference.get_detached_state_dict())).mean() + \
                        nn.ReLU()(1.0 + alpha * discriminator(cortex.generator.get_detached_state_dict())).mean()

            # train with gradient penalty
            gradient_penalty = discriminator.get_gradient_penalty(
                cortex.inference.get_detached_state_dict(),
                cortex.generator.get_detached_state_dict())
            disc_loss += gradient_penalty

        elif args.loss_type == 'wasserstein':
            disc_loss = -(alpha * discriminator(cortex.inference.get_detached_state_dict())).mean() + \
                         (alpha * discriminator(cortex.generator.get_detached_state_dict())).mean()

            # train with gradient penalty
            gradient_penalty = discriminator.get_gradient_penalty(
                cortex.inference.get_detached_state_dict(),
                cortex.generator.get_detached_state_dict())
            disc_loss += gradient_penalty
        else:
            d_inf = discriminator(cortex.inference.get_detached_state_dict())
            d_gen = discriminator(cortex.generator.get_detached_state_dict())

            p = .9 if args.label_smoothing else 1

            disc_loss = nn.BCELoss()(d_inf, p * Variable(torch.ones(batch_size, 1).to(real_samples.device))) + \
                        nn.BCELoss()(d_gen, (1-p) * Variable(torch.ones(batch_size, 1).to(real_samples.device)))

        if args.historical_averaging > 0:
            disc_loss = disc_loss + ha_loss(discriminator)

        # now update the inference and generator to fight the discriminator

        if args.loss_type == 'hinge' or args.loss_type == 'wasserstein':
            gen_loss = -discriminator(cortex.generator.intermediate_state_dict).mean() + \
                       discriminator(cortex.inference.intermediate_state_dict).mean()
        else:
            gen_loss = nn.BCELoss()(discriminator(cortex.generator.intermediate_state_dict),
                                    Variable(torch.ones(batch_size, 1).to(real_samples.device))) + \
                       nn.BCELoss()(discriminator(cortex.inference.intermediate_state_dict),
                                    Variable(torch.zeros(batch_size, 1).to(real_samples.device)))

        disc_loss.backward()
        optimizerD.step()

        gen_loss.backward()
        optimizerG.step()
        optimizerF.step()

        if not args.quiet:
            if batch % 100 == 0:
                print("Epoch {} Batch {} Overall surprisal {:.2f}".format(epoch, batch, ML_loss.item()))



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



    # NO so to use multiprocessing, the module we're calling it on needs to ever just be called as forward.'
    # No usage of calling its internal methods
    # The solution will be to multiprocessing wrap each of the methods we will ever call.

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


    cortex = DeterministicHelmholtz(args.noise_dim, args.n_filters, nc,
                                    image_size=image_size,
                                    surprisal_sigma=args.surprisal_sigma,
                                    log_intermediate_surprisals=args.detailed_logging,
                                    log_intermediate_reconstructions=args.detailed_logging,
                                    log_weight_alignment=args.detailed_logging,
                                    backprop_to_start_inf=bp_thru_inf,
                                    backprop_to_start_gen=bp_thru_gen)

    discriminator = Discriminator(image_size, args.disc_hidden_dim, cortex.layer_names,
                                  args.noise_dim, args.n_filters, nc,
                                  lambda_=args.lamda, loss_type=args.loss_type,
                                  log_intermediate_Ds=args.detailed_logging)

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

    optimizerD = optim.Adam(discriminator_params, lr=args.lr_d, betas=(args.beta1, 0.999), weight_decay = args.wd)
    optimizerG = optim.Adam(generator_params,     lr=args.lr_c, betas=(args.beta1, 0.999), weight_decay = args.wd)
    optimizerF = optim.Adam(inference_params,     lr=args.lr_c, betas=(args.beta1, 0.999), weight_decay = args.wd)

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

    for epoch in range(args.start_epoch, args.epochs):
        e = -1 if args.minimize_generator_surprisal else epoch + 1

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(args, cortex, train_loader, discriminator,
              optimizerD, optimizerG, optimizerF, epoch,
              ml_after_epoch = e)


        if args.save_imgs:
            try:
                os.mkdir("gen_images")
            except:
                pass
            to_visualize = cortex.noise_and_generate()[:100].detach().cpu()
            grid = utils.make_grid(to_visualize,
                                  nrow=10, padding=5, normalize=True,
                                  range=None, scale_each=False, pad_value=0)
            sv_img(grid, "gen_images/imgs_epoch{}.png".format(epoch), epoch)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            torch.save({
                'epoch': epoch + 1,
                'cortex_state_dict': cortex.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerF': optimizerF.state_dict(),
                'train_history': {"disc_loss" : discriminator.intermediate_Ds,
                                  "surprisals": cortex.intermediate_surprisals,
                                  "reconstructions": cortex.intermediate_reconstructions,
                                  "weight_alignment": cortex.weight_alignment}
            }, 'checkpoint.pth.tar')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
