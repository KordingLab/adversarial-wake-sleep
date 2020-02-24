from collections import OrderedDict
from scipy import linalg
import numpy as np
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('tanh1', nn.Tanh()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('tanh3', nn.Tanh()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('tanh5', nn.Tanh())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('tanh6', nn.Tanh()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

    def extract_features(self, img):
        output = self.convnet(img)
        output = output.view(img.size()[0],-1)
        output = self.fc[1](self.fc[0](output))
        return output

def get_distribution_of_labels_on_batch(imgs, net):
    net.eval()
    pad = torch.nn.ConstantPad2d(2, -.4242)
    feats = net(imgs)
    return feats


def get_marginalized_predicted_labels(cortex, lenet, n_batches=100):
    collected_stats = torch.ones(10, 1)
    for n in range(n_batches):
        imgs = cortex.noise_and_generate(4)
        feats = lenet(imgs)
        print(feats.size())
        collected_stats = collected_stats + feats


def extract_lenet_features(imgs, net):
    net.eval()
    pad = torch.nn.ConstantPad2d(2, -.4242)
    feats = net.extract_features(pad(imgs)).detach().cpu().numpy()
    return feats


def calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_mnist_fid(lenet, true_imgs, fake_imgs,
                        bootstrap=True, n_bootstraps=30):
    """Calculates the FID of the current generator

    inputs: lenet = loaded instance of LeNet
            true_imgs = a batch to input to LeNet

    Returns a tuple:
        score, standard_dev = scores is the score, standard_dev is from bootstrapping
            """
    act_true = extract_lenet_features(true_imgs, lenet)

    n_bootstraps = n_bootstraps if bootstrap else 1

    act_fake = extract_lenet_features(fake_imgs, lenet)

    fid_values = np.zeros((n_bootstraps))
    for i in range(n_bootstraps):
        act1_bs = act_true[np.random.choice(act_true.shape[0], act_true.shape[0], replace=True)]
        act2_bs = act_fake[np.random.choice(act_fake.shape[0], act_fake.shape[0], replace=True)]
        m1, s1 = calculate_activation_statistics(act1_bs)
        m2, s2 = calculate_activation_statistics(act2_bs)
        fid_values[i] = calculate_frechet_distance(m1, s1, m2, s2)
    results = (fid_values.mean(), fid_values.std())
    return results