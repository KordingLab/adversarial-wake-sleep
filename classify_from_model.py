"""Example call:
python ../classify_from_model.py --path checkpoint.pth.tar --gpu 1 --dataset cifar10 --data /home/abenjamin/data/ --n-filters 32 --noise-dim 100 --lr 1 --epochs 20 --wd 0.001 --opt sgd --lr-schedule"""

from torch import optim
from torchvision import datasets, transforms, utils
from torch.utils.data import Subset

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


from dcgan_models import DeterministicHelmholtz
from decoder_models import LinearDecoder, NonlinearDecoder

import argparse
import os
import random
import warnings



parser = argparse.ArgumentParser(description='Generate and save images of a network checkpoint.')
parser.add_argument('--path', metavar='DIR', default = "checkpoint.pth.tar",
                    help='path to the saved model checkpoint.')
parser.add_argument('-d', '--data', metavar='DIR', default = "../data",
                    help='path to dataset. Loads MNIST if nonexistant.')
parser.add_argument('--dataset', default='imagenet',
                    choices= ['imagenet', 'folder', 'lfw', 'cifar10', 'mnist'],
                    help="What dataset are we training on?")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1, type=float,
                    metavar='LRC', help='initial learning rate  Default 1e-4')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run. Default 50')
parser.add_argument('--opt', default='opt',
                    choices= ['adam','sgd'],
                    help="What algorithm to use?")
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='wd')



parser.add_argument('--noise-dim', default=40, type=int, metavar='ND',
                    help='Dimensionality of the top layer of the cortex.')
parser.add_argument('--n-filters', default=64, type=int,
                    help='Number of filters in the first conv layer of the DCGAN. Default 64')
parser.add_argument('--n-folds', default=10, type=int,
                    help='Number of CV folds to calculate the accuracy. Default 10.')
parser.add_argument('--nonlinear', action='store_true',
                    help="Don't use logistic regression but rather a 2-layer MLP")
parser.add_argument('--loss-type', default='wasserstein',
                    choices= ['BCE', 'wasserstein', 'hinge'],
                    help="The form of the minimax loss function. BCE = binary cross entropy of the original GAN")
parser.add_argument('--hidden-size', default=1000, type=int, metavar='ND',
                    help='Dimensionality of the hidden layer of the nonlinear decoder.')

parser.add_argument('--image-size', default=64, type=int,
                    help='Images to this many pixels. (default 64)')
parser.add_argument('--noise-type', default = 'none',
                    choices= ['none', 'fixed', 'learned_by_layer', 'learned_by_channel', 'learned_filter', 'poisson'],
                    help="What variance of Gaussian noise should be applied after all layers in the "
                            "cortex? See docs for details. Default is no noise; fixed has variance 0.01")
parser.add_argument('--he-initialization', action='store_true',
                    help='As in ProgressiveGANs. Plays well with divisive normalization')
parser.add_argument('--divisive-normalization', action='store_true',
                    help='Divisive normalization over channels, pixel by pixel. As in ProgressiveGANs')
parser.add_argument('--lr-schedule', action='store_true',
                    help='Learning rate *=.1 halfway through.')

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
        warnings.warn('You have chosen a specific GPU. ')

    cortex = load_cortex(args.path, args)

    accuracies =   decode_classes_from_layers(args.gpu,
                                              cortex,
                                              args.image_size,
                                              args.n_filters,
                                              args.noise_dim,
                                              args.data,
                                              args.dataset,
                                              args.nonlinear,
                                              args.lr,
                                              args.n_folds,
                                              args.epochs,
                                              args.hidden_size,
                                              args.wd,
                                              args.opt,
                                              args.lr_schedule,
                                              args.batch_size,
                                              args.workers)

    for i in range(6):
        print("Layer{}: Accuracy {} +/- {}".format(i, accuracies.mean(dim=0)[i],accuracies.std(dim=0)[i]))

def load_cortex(path, args):
    """Loads a cortex from path."""
    bn = False if args.loss_type == 'wasserstein' else True

    cortex = DeterministicHelmholtz(args.noise_dim, args.n_filters,
                                    1 if args.dataset == 'mnist' else 3,
                                    image_size=args.image_size,
                                    noise_type = args.noise_type,
                                    batchnorm = bn,
                                    normalize = args.divisive_normalization,
                                    he_init=args.he_initialization)

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        # load onto the CPU
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        cortex.load_state_dict(checkpoint['cortex_state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        raise IOError("=> no checkpoint found at '{}'".format(path))

    return cortex


def train(cortex, optimizer, decoder, train_loader, gpu):
    """Given some training data, feed that through cortex, put all activations through decoders,
    and train the decoder on the supervised task"""
    cortex.eval()
    decoder.train()

    loss_fn = nn.CrossEntropyLoss()

    for batch, labels in train_loader:
        optimizer.zero_grad()
        batch, labels = batch.cuda(gpu), labels.cuda(gpu)

        # run though
        cortex.inference(batch)

        # get predictions
        predictions = decoder(cortex.inference.intermediate_state_dict)

        loss = 0
        for pred in predictions:
            loss = loss + loss_fn(pred, labels)
        loss.backward()
        optimizer.step()

def test(cortex, decoder, test_loader, gpu, epoch, n_examples, verbose = False):
    """Returns the classification error and loss on this fold of the test set for each of the 5 layers + the input"""
    cortex.eval()
    decoder.eval()

    correct = [0 for _ in range(6)]
    total = 0
    for batch, labels in test_loader:
        batch, labels = batch.cuda(gpu), labels.cuda(gpu)
        # run though
        cortex.inference(batch)

        # get predictions
        predictions = decoder(cortex.inference.intermediate_state_dict)

        # get accuracy on each layer
        total += labels.size(0)
        for i in range(6):
            _, predicted = torch.max(predictions[i].data, 1)
            correct[i] += float((predicted == labels).sum().item())

    accuracies = []
    if verbose:
        print('Epoch {}: accuracy on {} test images:'.format(epoch, n_examples))
    for i in range(6):
        accuracy = 100 * correct[i] / total
        accuracies.append(accuracy)
        if verbose:
            print('Layer{}: {}'.format(i, accuracy))

    return accuracies

def decode_classes_from_layers(gpu,
                              cortex,
                              image_size,
                              n_filters,
                              noise_dim,
                              data_path,
                              dataset,
                              nonlinear = False,
                              lr = 0.001,
                              folds = 10,
                              epochs = 50,
                              hidden_size = 1000,
                              wd = 1e-4,
                              opt = 'adam',
                              lr_schedule = False,
                              batch_size = 128,
                              workers = 4,
                              verbose = True):

    """ Trains a linear or nonlinear decoder from a given layer of the cortex, all layers at a time (including inputs)

    Does k-fold CV on the test set of this dataset. A random permutation is used.

    Returns a tensor of accuracies on each of the k folds and for each of the 6 decoders:
                                                        Input ---  Layer1 .... Layer4 --- Noise


   """

    # ----- Get dataset ------ #
    # Data loading code
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        all_test_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]))

        nc = 3
        n_classes = 1000
    elif dataset == 'cifar10':
        all_test_dataset = datasets.CIFAR10(root=data_path, download=True, train=False,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc = 3
        n_classes = 10
    elif dataset == 'mnist':
        all_test_dataset = datasets.MNIST(root=data_path, download=True, train=False,
                             transform=transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
        nc = 1
        n_classes = 10

    assert all_test_dataset

    perm = torch.randperm(len(all_test_dataset))
    n_test_examples = len(all_test_dataset) // folds

    all_accuracies = []
    for f in range(folds):
        # ---- Get CV indices ----
        test_idx = perm[f * n_test_examples: (f+1) * n_test_examples]
        if f==folds-1:
            #last fold may be larger if len(all_test_dataset) % folds != 0
            test_idx = perm[f * n_test_examples:]

        train_idx = torch.cat((perm[:f * n_test_examples],
                              perm[(f + 1) * n_test_examples:]))

        #  ----- Make loaders -----
        train_dataset = Subset(all_test_dataset, train_idx)
        test_dataset = Subset(all_test_dataset, test_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,)

        # ----- Build decoder ------
        if nonlinear:
            decoder = NonlinearDecoder(image_size, noise_dim, n_classes, nc, n_filters, hidden_size)
        else:
            decoder = LinearDecoder(image_size, noise_dim, n_classes, nc, n_filters)

        # get to proper GPU
        torch.cuda.set_device(gpu)
        cortex = cortex.cuda(gpu)
        decoder = decoder.cuda(gpu)

        # ------ Build optimizer ------ #

        if opt == 'adam':
            optimizer = optim.Adam(decoder.parameters(), lr=lr, betas=(.9, 0.999), weight_decay = wd)
        elif opt == 'sgd':
            optimizer = optim.SGD(decoder.parameters(), lr=lr, momentum = 0.9, weight_decay = wd)
        else:
            raise AssertionError("This optimizer not implemented yet.")

        for epoch in range(epochs):
            if lr_schedule:
                adjust_lr(epoch, optimizer, epochs)
            train(cortex, optimizer, decoder, train_loader, gpu)
            if verbose or (epoch==epochs-1):
                accuracies = test(cortex, decoder, test_loader, gpu, epoch, len(test_idx), verbose)

        all_accuracies.append(accuracies)

    return torch.Tensor(all_accuracies)

def adjust_lr(epoch, optimizer,epochs):
    if epoch %(epochs//3)==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= .3

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
