"""Example call:
python ../classify_from_model.py --path checkpoint.pth.tar --gpu 1 --dataset cifar10 --data /home/abenjamin/data/ --n-filters 32 --noise-dim 100 --lr 1 --epochs 20 --wd 0.001 --opt sgd --lr-schedule"""

from torch import optim
from torchvision import datasets, transforms, utils
from torch.utils.data import Subset

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import  numpy as np

from dcgan_models import Inference, Generator
from decoder_models import LinearDecoder, NonlinearDecoder
from utils import gen_surprisal

import argparse
import os
import random
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


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

parser.add_argument('--alpha', default=1e-3, type=float,
                    help='svm regularization',
                   )


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

    inference = load_cortex(args.path, args)

    accuracies =  np.array(decode_classes_from_layers(args.gpu,
                                          inference,
                                          args.image_size,
                                          args.data,
                                          args.dataset,
                                          folds = args.n_folds,
                                          epochs = args.epochs,
                                          batch_size = args.batch_size,
                                          workers = args.workers))
    for i in range(6):
        print("Layer{}: Accuracy {} +/- {}".format(i, accuracies.mean(0)[i],accuracies.std(0)[i]))

def load_cortex(path, args):
    """Loads a cortex from path."""

    inference = Inference(args.noise_dim, args.n_filters,
                        1 if args.dataset == 'mnist' else 3,
                        image_size=args.image_size,
                       bn=True, hard_norm=args.divisive_normalization, spec_norm=False, derelu=True)


    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        # load onto the CPU
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        inference.load_state_dict(checkpoint['inference_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        raise IOError("=> no checkpoint found at '{}'".format(path))

    return inference


def train(inference, decoders,scalers, train_loader, gpu):
    """Given some training data, feed that through cortex, put all activations through decoders,
    and train the decoder on the supervised task"""
    inference.eval()

    for batch, labels in train_loader:
        batch = batch.cuda(gpu)

        # run though
        inference(batch)

        for i, (decoder, scaler) in enumerate(zip(decoders, scalers)):
            layer_name = inference.layer_names[i]
            data = inference.intermediate_state_dict[layer_name].cpu().detach().view(batch.size(0), -1)
            scaler.partial_fit(data)
            decoder.partial_fit(scaler.transform(data), labels, classes = range(10))



def test(inference, decoders, scalers,test_loader, gpu):
    """Returns the classification error and loss on this fold of the test set for each of the 5 layers + the input"""
    inference.eval()

    correct = [0 for _ in range(6)]
    n = 0

    for batch, labels in test_loader:
        batch = batch.cuda(gpu)
        # run though
        inference(batch)

        # get predictions
        for i, (decoder, scaler) in enumerate(zip(decoders, scalers)):
            layer_name = inference.layer_names[i]
            data = inference.intermediate_state_dict[layer_name].cpu().detach().view(batch.size(0),-1)
            predictions = decoder.predict(scaler.transform(data))

            n_right = (predictions==labels.numpy()).sum()
            correct[i] += n_right
        n += int(batch.size(0))

    accuracies = np.array(correct)/(float(n))

    return accuracies

def decode_classes_from_layers(gpu,
                              inference,
                              image_size,
                              data_path,
                              dataset,
                              folds = 10,
                              epochs = 50,
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

    accuracies = []
    for f in range(folds):
        if verbose:
            print("Fold {}".format(f))
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
        decoders = [SGDClassifier(max_iter=1000, tol=1e-3, alpha=args.alpha, penalty='elasticnet', n_jobs = 6,
                                  learning_rate='optimal') for _ in range(6)]
        scalers = [StandardScaler() for _ in range(6)]

        # get to proper GPU
        torch.cuda.set_device(gpu)
        inference = inference.cuda(gpu)


        for epoch in range(epochs):

            train(inference, decoders, scalers, train_loader, gpu)
            if verbose or (epoch==epochs-1):
                accuracy = test(inference, decoders,scalers, test_loader, gpu)
                print("Epoch {} Accuracies {}".format(epoch, accuracy))
        accuracies.append(accuracy)
    return accuracies

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
