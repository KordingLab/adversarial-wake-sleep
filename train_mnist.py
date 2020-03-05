from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils

import torch
import torch.nn as nn

from train_utils_mnist import engage_new_layer, get_batch_of_real, generate_some_images
from mnist_fid import LeNet5, calculate_mnist_fid
from mnist_models import DeterministicHelmholtz, Discriminator
from utils import sv_img

import argparse
import os



parser = argparse.ArgumentParser(description='PyTorch Adversarial Wake-Sleep Training on MNIST')
parser.add_argument('-d', '--data', metavar='DIR', default = "../data",
                    help='path to dataset. Loads MNIST if nonexistant.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr-c', '--learning-rate-cortex', default=1e-4, type=float,
                    metavar='LRC', help='initial learning rate of the inference and generator', dest='lr_c')
parser.add_argument('--lr-d', '--learning-rate-discriminator', default=5e-4, type=float,
                    metavar='LRD', help='initial learning rate of the discriminator', dest='lr_d')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='wd')
parser.add_argument('--beta1',  default=.5, type=float,
                    help='In the adam optimizer. Default = 0.5')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run. Default 90')
parser.add_argument('--sequential-training', action='store_true',
                    help='Whether to add layers one by one when training.')
parser.add_argument( '--loss-type', default='BCE',
                    choices= ['BCE', 'wasserstein', 'hinge'],
                    help="The form of the minimax loss function. BCE = binary cross entropy of the original GAN")
parser.add_argument('--detailed-logging', action='store_true',
                    help='Whether to store detailed information about training metrics inside of the cortex object')
parser.add_argument('--noise-dim', default=40, type=int, metavar='ND',
                    help='Dimensionality of the top layer of the cortex.')
parser.add_argument('--disc-hidden-dim', default=256, type=int,
                    help='Dimensionality of the hidden layer in each discriminator.')
parser.add_argument('--surprisal-sigma', default=1, type=float,
                    metavar='ss', help='How weakly to follow the gradient of the surprisal of the lower layers '
                     'given the upper layers inference activations and the generative model.'
                     ' Specifically the variance of the implied Gaussian noise of the output (but there is no noise.'
             ' Equivalent to minimizing the reconstruction error from inference states, up one layer, and back down.')
parser.add_argument('--minimize-generator-surprisal', action='store_true',
                    help='Minimize generator surprisal using value of sigma set.')
parser.add_argument('--lamda', default=.1, type=float,
                    help='Lambda for the gradient penalty in the WGAN formulation. Only for Wasserstein loss.')
parser.add_argument('--no-backprop-through-full-cortex', action='store_true',
                    help='Only backprop through the local (layerwise) discriminator to parameters in that same '
                     'layer of the cortex. For biological plausibility.')
parser.add_argument('--only-backprop-generator', action='store_true',
                    help='Only backprop through the local (layerwise) discriminator to parameters in that same '
                     'layer of the cortex for the inference. For the generator we backprop all the way to the top.')
parser.add_argument('--label-smoothing', action='store_true',
                    help='Make the discriminator be less confident by randomly switching labels with p=.1')
parser.add_argument('--noise-sigma', default=0, type=float,
                    help='If set, add Gaussian noise with this variance to the pre-Relu activations of both passes.')



# printing and saving
parser.add_argument('--print-fid', action='store_true',
                    help='Calculate and return the FID score on MNIST using a pretrained LeNet')
parser.add_argument('--quiet', action='store_true',
                    help='Do not print the surprisal stats.')
parser.add_argument("--savename",
                        help = "Relative path to a folde-dr in which to save the model, discriminator, and history",
                        type = str,
                        default = './cortex_disc_and_development.pickle')
parser.add_argument('--save-imgs', action='store_true',
                    help='Save the generated images every epoch')



def train(args, cortex, train_loader, discriminator, train_history,
          optimizerD, optimizerG, optimizerF, epoch, ml_after_epoch=-1):
    if args.sequential_training:
        noise_layer = engage_new_layer(epoch, cortex, optimizerG, optimizerF, optimizerD, discriminator,
                                       n_epochs_per_layer=25)
    else:
        noise_layer = len(cortex.generator_modules)

    for batch, (images, _) in enumerate(train_loader):
        cortex.log_weight_alignment()

        batch_size = images.size()[0]

        optimizerG.zero_grad()
        optimizerD.zero_grad()
        optimizerF.zero_grad()

        ############### WAKE #################
        # get some data
        real_samples = Variable(images).cuda(args.gpu)

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

        # now update the inference and generator to fight the discriminator

        if args.loss_type == 'hinge' or args.loss_type == 'wasserstein':
            gen_loss = -discriminator(cortex.generator.intermediate_state_dict).mean() + \
                       discriminator(cortex.inference.intermediate_state_dict).mean()
        else:
            gen_loss = nn.BCELoss()(discriminator(cortex.generator.intermediate_state_dict),
                                    Variable(torch.ones(batch_size, 1).cuda(args.gpu))) + \
                       nn.BCELoss()(discriminator(cortex.inference.intermediate_state_dict),
                                    Variable(torch.zeros(batch_size, 1).cuda(args.gpu)))

        disc_loss.backward()
        optimizerD.step()

        gen_loss.backward()
        optimizerG.step()
        optimizerF.step()

        if batch % 47 == 0:
            # get reconstruction loss to measure F
            reconstruction_MSE = cortex.mse(real_samples, cortex.generator(inferred_noise))
            train_history['reconstruction_error'].append(reconstruction_MSE.item())

            train_history['D_losses'].append(disc_loss.item())
            train_history['ML_losses'].append(ML_loss.item())
            train_history['GF_losses'].append(gen_loss.item())

            if not args.quiet:
                print("Epoch {} Batch {} Overall surprisal {:.2f}".format(epoch, batch, ML_loss.item()))


def main(args):

    torch.cuda.set_device(args.gpu)

    mnist_size = (1, 28, 28)

    # False if no bp thru cortex, or if only bp gen, else true
    bp_thru_inf = (not args.no_backprop_through_full_cortex) and (not args.only_backprop_generator)
    bp_thru_gen = (not args.no_backprop_through_full_cortex)

    cortex = DeterministicHelmholtz(image_size=mnist_size, noise_dim=args.noise_dim, surprisal_sigma=args.surprisal_sigma,
                                    log_intermediate_surprisals=args.detailed_logging,
                                    log_intermediate_reconstructions=args.detailed_logging,
                                    log_weight_alignment=args.detailed_logging,
                                    noise_sigma = args.noise_sigma,
                                    backprop_to_start_inf=bp_thru_inf,
                                    backprop_to_start_gen=bp_thru_gen).cuda(args.gpu)

    discriminator = Discriminator(cortex.layer_names, lambda_=args.lamda, loss_type=args.loss_type,
                                  noise_dim=args.noise_dim,
                                  log_intermediate_Ds=args.detailed_logging).cuda(args.gpu)

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    train_history = {'D_losses': [],
                     'GF_losses': [],
                     'ML_losses': [],
                     'reconstruction_error': [],
                     }
    if args.print_fid:
        lenet = LeNet5().cuda(args.gpu)
        lenet.load_state_dict(torch.load('/home/abenjamin/wakesleep/gan-metrics-pytorch/models/lenet.pth'))

    if args.sequential_training:
        generator_params = cortex.generator.generative_1to0_conv.parameters()
        inference_params = cortex.inference.inference_0to1_conv.parameters()
        discriminator_params = discriminator.discriminator_0and1.parameters()
    else:
        generator_params = cortex.generator.parameters()
        inference_params = cortex.inference.parameters()
        discriminator_params = discriminator.parameters()

    # set up optimizer
    optimizerD = optim.Adam(discriminator_params, lr=args.lr_d, betas=(args.beta1, 0.999), weight_decay = args.wd)
    optimizerG = optim.Adam(generator_params,     lr=args.lr_c, betas=(args.beta1, 0.999), weight_decay = args.wd)
    optimizerF = optim.Adam(inference_params,     lr=args.lr_c, betas=(args.beta1, 0.999), weight_decay = args.wd)

    for epoch in range(args.epochs):
        e = -1 if args.minimize_generator_surprisal else epoch + 1

        train(args, cortex, train_loader, discriminator, train_history,
              optimizerD, optimizerG, optimizerF, epoch,
              ml_after_epoch=e)

        if args.print_fid:

            some_fake_images = generate_some_images(cortex, args.noise_dim, 512)
            some_real_imgs = get_batch_of_real(test_loader)
            fid_score, conf_int = calculate_mnist_fid(lenet, some_real_imgs, some_fake_images,
                                                      bootstrap=True)
            print("Epoch {} FID score of {} +/- {}".format(epoch, fid_score, conf_int))

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

        torch.save({
            'epoch': epoch + 1,
            'cortex_state_dict': cortex.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerF': optimizerF.state_dict(),
            'train_history': {"disc_loss": discriminator.intermediate_Ds,
                              "surprisals": cortex.intermediate_surprisals,
                              "reconstructions": cortex.intermediate_reconstructions,
                              "weight_alignment": cortex.weight_alignment}
        }, 'checkpoint.pth.tar')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)