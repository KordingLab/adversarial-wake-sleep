import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions import Laplace, Normal
from torch.autograd import Variable, grad
import math
from numpy import prod

class Generator(nn.Module):
    """ The feedback edges of the cortex. Generates images with the DCGAN architecture.
    
    Parameters
    noise_dim
    n_filters
    n_img_channels
    noise_type: What sort of noise is applied after each convolutional layer? Gaussian, but of what variance?
                'fixed' = Noise is always Gaussian with variance 0.01
                'none' = No noise
                'learned_by_layer' = The variance is learned and different for each layer
                'learned_by_channel' = The variance is learned and different for each channel and each layer
                'learned_filter' = Tariance that is the result of a learned filter
                                            on the previous layer. Like the `reparameterization trick` of 
                                            variational autoencoders.
                'poisson' = variance is equal to value
    backprop_to_start
    image_size
    batchnorm
    normalize
    he_init
    
    """
    def __init__(self, noise_dim, n_filters, n_img_channels, noise_type = 'none', backprop_to_start = True,
                 image_size = 64, batchnorm = False, normalize = False, he_init = False):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.n_filters = n_filters
        self.n_img_channels = n_img_channels
        self.backprop_to_start = backprop_to_start

        # A small note regarding he initialization (if used) for those curious:
        # An interesting thing with the ConvTranspose is that the number of effective inputs is not
        # kernel_size ** 2 * in_channels, but rather depends on the stride and size (due to expansion
        # of the inputs and convolving with padding; both implicitly apply 0s)

        # In the learned_filter mode, all conv layers need to output twice the number of channels as before.
        maybetimestwo = 2 if noise_type=="learned_filter" else 1

        self.generative_5to4_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_dim, n_filters * 8 * maybetimestwo, image_size//16, 1, 0 ),
            MaybeHeRescale(noise_dim, 1,rescale=he_init),
            AddNoise(noise_type, n_filters * 8),
            nn.BatchNorm2d(n_filters * 8) if batchnorm else null(),
            nn.ReLU(),
            NormalizationLayer() if normalize else null())

        self.generative_4to3_conv = nn.Sequential(
            # state size. (n_filters*8) x 4 x 4
            nn.ConvTranspose2d(n_filters * 8, n_filters * 4 * maybetimestwo, 4, 2, 1 ),
            MaybeHeRescale(n_filters * 8, 2, rescale=he_init),
            AddNoise(noise_type, n_filters * 4),
            nn.BatchNorm2d(n_filters * 4) if batchnorm else null(),
            nn.ReLU(),
            NormalizationLayer() if normalize else null())

        self.generative_3to2_conv = nn.Sequential(
            # state size. (n_filters*4) x 8 x 8
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2 * maybetimestwo, 4, 2, 1 ),
            MaybeHeRescale(n_filters * 4, 2, rescale=he_init),
            AddNoise(noise_type, n_filters * 2),
            nn.BatchNorm2d(n_filters * 2) if batchnorm else null(),
            nn.ReLU(),
            NormalizationLayer() if normalize else null())

        self.generative_2to1_conv = nn.Sequential(
            # state size. (n_filters*2) x 16 x 16
            nn.ConvTranspose2d(n_filters * 2, n_filters * maybetimestwo, 4, 2, 1 ),
            MaybeHeRescale(n_filters * 2, 2, rescale=he_init),
            AddNoise(noise_type, n_filters),
            nn.BatchNorm2d(n_filters) if batchnorm else null(),
            nn.ReLU(),
            NormalizationLayer() if normalize else null())
            # state size. (n_filters) x 32 x 32

        self.generative_1to0_conv = nn.Sequential(
            nn.ConvTranspose2d(n_filters, n_img_channels * maybetimestwo, 4, 2, 1 ),
            MaybeHeRescale(n_filters, 2, rescale=he_init),
            AddNoise(noise_type, n_img_channels),
            nn.Tanh()
            # state size. (n_img_channels) x 64 x 64
        )

        # list modules bottom to top. Probably a more general way exists
        self.listed_modules = [self.generative_1to0_conv,
                              self.generative_2to1_conv,
                              self.generative_3to2_conv,
                              self.generative_4to3_conv,
                              self.generative_5to4_conv]

        self.intermediate_state_dict = OrderedDict([('Input',  None),
                                                    ('Layer1', None),
                                                    ('Layer2', None),
                                                    ('Layer3', None),
                                                    ('Layer4', None),
                                                    ('Layer5', None)])
        self.layer_names = list(self.intermediate_state_dict.keys())



    def get_detached_state_dict(self):
        detached_dict = {k: None if (v is None) else v.detach()
                            for k, v in self.intermediate_state_dict.items()}
        return detached_dict

    def forward(self, x, from_layer = 5):
        # iterate through layers and pass the noise downwards
        for i, (G, layer_name) in enumerate(zip(self.listed_modules[::-1],
                                                self.layer_names[:0:-1])):
            # Skip the topmost n layers ?
            if len(self.listed_modules) - i > from_layer:
                continue

            self.intermediate_state_dict[layer_name] = x

            # this setting makes all gradient flow only go one layer back
            if not self.backprop_to_start:
                x = x.detach()
            x = G(x)

        self.intermediate_state_dict["Input"] = x

        return x


class Inference(nn.Module):
    def __init__(self, noise_dim, n_filters, n_img_channels, noise_type = 'none', backprop_to_start = True,
                 image_size = 64, batchnorm = False, normalize = False, he_init = False):
        super(Inference, self).__init__()
        self.noise_dim = noise_dim
        self.n_filters = n_filters
        self.n_img_channels = n_img_channels
        self.backprop_to_start = backprop_to_start

        # In the learned_filter mode, all conv layers need to output twice the number of channels as before.
        maybetimestwo = 2 if noise_type=="learned_filter" else 1


        self.inference_4to5_conv = nn.Sequential(
            nn.Conv2d(n_filters * 8, noise_dim * maybetimestwo, image_size // 16, 1, 0 ),
            MaybeHeRescale(n_filters * 8, image_size // 16, rescale=he_init),
            AddNoise(noise_type, noise_dim)
        )

        self.inference_3to4_conv = nn.Sequential(
            nn.Conv2d(n_filters * 4, n_filters * 8 * maybetimestwo, 4, 2, 1 ),
            MaybeHeRescale(n_filters * 4, 4, rescale=he_init),
            AddNoise(noise_type, n_filters * 8),
            nn.BatchNorm2d(n_filters * 8) if batchnorm else null(),
            nn.ReLU(),
            NormalizationLayer() if normalize else null())

        self.inference_2to3_conv = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters * 4 * maybetimestwo, 4, 2, 1 ),
            MaybeHeRescale(n_filters * 2, 4, rescale=he_init),
            AddNoise(noise_type, n_filters * 4),
            nn.BatchNorm2d(n_filters * 4) if batchnorm else null(),
            nn.ReLU(),
            NormalizationLayer() if normalize else null())

        self.inference_1to2_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 2 * maybetimestwo, 4, 2, 1 ),
            MaybeHeRescale(n_filters, 4, rescale=he_init),
            AddNoise(noise_type, n_filters * 2),
            nn.BatchNorm2d(n_filters * 2) if batchnorm else null(),
            nn.ReLU(),
            NormalizationLayer() if normalize else null())

        self.inference_0to1_conv = nn.Sequential(
            nn.Conv2d(n_img_channels, n_filters * maybetimestwo, 4, 2, 1 ),
            MaybeHeRescale(n_img_channels, 4, rescale=he_init),
            AddNoise(noise_type, n_filters),
            nn.BatchNorm2d(n_filters) if batchnorm else null(),
            nn.ReLU(),
            NormalizationLayer() if normalize else null())


        # list modules bottom to top. Probably a more general way exists
        self.listed_modules = [self.inference_0to1_conv,
                               self.inference_1to2_conv,
                               self.inference_2to3_conv,
                               self.inference_3to4_conv,
                               self.inference_4to5_conv]

        self.intermediate_state_dict = OrderedDict([('Input', None),
                                                    ('Layer1', None),
                                                    ('Layer2', None),
                                                    ('Layer3', None),
                                                    ('Layer4', None),
                                                    ('Layer5', None)])
        self.layer_names = list(self.intermediate_state_dict.keys())

    def get_detached_state_dict(self):
        detached_dict = {k: None if (v is None) else v.detach()
                         for k, v in self.intermediate_state_dict.items()}
        return detached_dict

    def forward(self, x):
        self.intermediate_state_dict['Input'] = x
        # iterate through layers and pass the input upwards
        for F, layer_name in zip(self.listed_modules,
                                 self.layer_names[1:]):
            # this setting makes all gradient flow only go one layer back
            if not self.backprop_to_start:
                x = x.detach()

            # print(layer_name, x.size())
            x = F(x)
            self.intermediate_state_dict[layer_name] = x

        return x


class DeterministicHelmholtz(nn.Module):
    """

    input_size = size of the input images. Result of images.size()
    noise_dim = dimension of the noise vector at the highest level
    surprisal_sigma = how strongly to follow the gradient of layer-wise surprisal
                        (or, variance of the gaussian distribution asserted when comparing predicted
                            vs. actual lower-layer activity)

    """

    def __init__(self, noise_dim, n_filters, n_img_channels,
                 image_size = 64,
                 surprisal_sigma=1.0,
                 noise_type = None,
                 detailed_logging = True,
                 backprop_to_start_inf=True,
                 backprop_to_start_gen=True,
                 batchnorm = False,
                 normalize=False,
                 he_init = False):
        super(DeterministicHelmholtz, self).__init__()

        assert image_size % 16 == 0

        self.inference = Inference(noise_dim, n_filters, n_img_channels, noise_type, backprop_to_start_inf, image_size,
                                   batchnorm=batchnorm, normalize=normalize, he_init = he_init)
        self.generator = Generator(noise_dim, n_filters, n_img_channels, noise_type, backprop_to_start_gen, image_size,
                                   batchnorm=batchnorm, normalize=normalize, he_init = he_init)

        # init
        if he_init:
            self.inference.apply(weights_init_he)
            self.generator.apply(weights_init_he)
        else:
            self.inference.apply(weights_init)
            self.generator.apply(weights_init)

        self.mse = nn.MSELoss()
        self.surprisal_sigma = surprisal_sigma

        self.layer_names = list(self.inference.intermediate_state_dict.keys())
        self.which_layers = range(len(self.layer_names))
        self.noise_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        # logging
        self.log_intermediate_surprisals = detailed_logging
        self.intermediate_surprisals = {layer: [] for layer in self.layer_names}

        self.log_intermediate_reconstructions = detailed_logging
        self.intermediate_reconstructions = {layer: [] for layer in self.layer_names}

        self.log_weight_alignment_ = detailed_logging
        self.weight_alignment = {layer: [] for layer in self.layer_names}

        self.log_channel_norms = detailed_logging
        self.channel_norms = {layer: [] for layer in self.layer_names}


    def infer(self, x):
        return self.inference(x)

    def generate(self, x):
        return self.generator(x)

    def noise_and_generate(self, noise_layer = 5):
        """Sample some noise at a given layer and propagate to the bottom.

        Noise_layer = int, with 0 being input and 5 being the very top
        """

        if self.inference.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        noise_layer_str = list(self.layer_names)[noise_layer]

        # Sample some noise at this layer
        x = self.noise_dist.sample(self.inference.intermediate_state_dict[noise_layer_str].size()).to(
            self.inference.intermediate_state_dict[noise_layer_str].device)
        x = x.squeeze(dim=-1)
        # x = torch.abs(x)

        x = self.generator(x, from_layer = noise_layer)

        return x

    def generator_surprisal(self):
        """Given the current inference state, ascertain how surprised the generator model was.

        """
        # here we have to a assume a noise model in order to calculate p(h_1 | h_2 ; G)
        # with Gaussian we have log p  = MSE between actual and predicted

        if self.inference.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        ML_loss = 0

        for i, G in enumerate(self.generator.listed_modules):
            if i not in self.which_layers and not self.log_intermediate_surprisals:
                continue

            lower_h = self.inference.intermediate_state_dict[self.layer_names[i]].detach()
            upper_h = self.inference.intermediate_state_dict[self.layer_names[i + 1]].detach()

            x = G(upper_h)

            layerwise_surprisal = self.mse(x, lower_h)

            if i in self.which_layers:
                ML_loss = ML_loss + layerwise_surprisal

            if self.log_intermediate_surprisals:
                self.intermediate_surprisals[self.layer_names[i]].append(layerwise_surprisal.item())

        ML_loss = ML_loss / self.surprisal_sigma

        return ML_loss

    def inference_surprisal(self):
        """Given the current inference state, ascertain how surprised the generator model was.
        Equivalent to minimizing the reconstruction error (i->i-1->i) during generation.
        """
        # here we have to a assume a noise model in order to calculate p(h_1 | h_2 ; G)
        # with Gaussian we have log p  = MSE between actual and predicted

        if self.generator.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        ML_loss = 0

        for i, F in enumerate(self.inference.listed_modules):
            if i not in self.which_layers:
                continue

            lower_h = self.generator.intermediate_state_dict[self.layer_names[i]].detach()
            upper_h = self.generator.intermediate_state_dict[self.layer_names[i + 1]].detach()

            F_upper_h = F(lower_h)

            layerwise_surprisal = self.mse(upper_h, F_upper_h)
            if i in self.which_layers:
                ML_loss = ML_loss + layerwise_surprisal

        ML_loss = ML_loss / self.surprisal_sigma

        return ML_loss

    def get_pixelwise_channel_norms(self):
        """This can be used to implement a 'soft' divisive normalization (where the 'hard' version is
        that which is implemented in Karras et al.). It can also be used for logging & debugging.

        Logs the total value by layer as a tuple (inferred norm, generated norm)

        Returns the pixel-wise distance from 0 (as a cost) to be minimized)"""

        if self.inference.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")
        if self.generator.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        epsilon = 1e-8

        total_distance_from_1 = 0

        for i, layer in enumerate(self.layer_names):

            inferred_state = self.inference.intermediate_state_dict[layer]
            generated_state = self.generator.intermediate_state_dict[layer]

            inferred_state_pixel_norm = (((inferred_state**2).mean(dim=1) + epsilon).sqrt())
            inferred_state_pixel_norm_dist_from_1 = self.mse(inferred_state_pixel_norm,
                                                             torch.ones_like(inferred_state_pixel_norm))
            total_distance_from_1 = inferred_state_pixel_norm_dist_from_1 + total_distance_from_1

            generated_state_pixel_norm = (((generated_state**2).mean(dim=1) + epsilon).sqrt())
            generated_state_pixel_norm_dist_from_1 = self.mse(generated_state_pixel_norm,
                                                             torch.ones_like(generated_state_pixel_norm))
            total_distance_from_1 = generated_state_pixel_norm_dist_from_1 + total_distance_from_1

            if self.log_channel_norms:

                self.channel_norms[layer].append((inferred_state_pixel_norm.mean().item(),
                                                 generated_state_pixel_norm.mean().item()))

        return total_distance_from_1


    def log_layerwise_reconstructions(self):
        """From the current *inferential* distribution, determine (and record?) the
        error upon reconstruction at each layer (i.e. generating down and inferring back up).

        For debugging purposes only; we never backprop through this."""

        if self.inference.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        if self.log_intermediate_reconstructions:
            for i, (G, F) in enumerate(zip(self.generator.listed_modules, self.inference.listed_modules)):
                if i not in self.which_layers:
                    self.intermediate_reconstructions[self.layer_names[i]].append(-1)
                    continue

                lower_h = self.inference.intermediate_state_dict[self.layer_names[i]].detach()

                # if lower_h.is_cuda and self.ngpu > 1:
                #     x = nn.parallel.data_parallel(F, lower_h, range(self.ngpu))
                #     reconstruction = nn.parallel.data_parallel(G, x, range(self.ngpu))
                # else:
                x = F(lower_h)
                reconstruction = G(x)


                error = self.mse(lower_h, reconstruction)
                self.intermediate_reconstructions[self.layer_names[i]].append(error.item())

    def log_weight_alignment(self):
        if self.log_weight_alignment_:
            for i, (G, F) in enumerate(zip(self.generator.listed_modules, self.inference.listed_modules)):
                gen_weight = list(G.parameters())[0]
                inf_weight = list(F.parameters())[0]

                cosine = torch.nn.CosineSimilarity()(inf_weight.transpose(1, 0).cpu().view(1, -1),
                                                     gen_weight.cpu().view(1, -1))
                angle = torch.acos(cosine).item()
                self.weight_alignment[self.layer_names[i]].append(angle)

class null(nn.Module):
    "Pickleable nothing"
    def __init__(self):
        super(null, self).__init__()

    def forward(self, x):
        return x

class Discriminator(nn.Module):
    """To discriminate between full network states.
    The network hidden states are concatenated

    x     h1     h2     h3     h4     h5
    |     |      |      |      |      |
    +-c1--+--c2--+--c3--+--c4--+--c5--+--c6--> output


    n_filters = number of filters in the first layer of E/G. assumes this doubles each layer
    n_img_channels = number of color channels in the input
    n_noise_channels = dimensionality of h5

    """

    def __init__(self, n_filters, n_img_channels, n_noise_channels, image_size,
                eval_std_dev = False,
                log_channel_norms = False,
                detailed_logging = False,
                 lambda_ = .5):
        super(Discriminator, self).__init__()

        self.relu = nn.LeakyReLU(.2, inplace = True)


        self.eval_std_dev = eval_std_dev
        plus1 = 1 if eval_std_dev else 0

        last_kernel = 4 if image_size==64 else 2


        self.layers = nn.ModuleList([nn.Conv2d(n_img_channels, n_filters, 4, 2, 1),
                                     nn.Conv2d(n_filters*2, n_filters*2, 4, 2, 1),
                                     nn.Conv2d(n_filters*4, n_filters*4, 4, 2, 1),
                                     nn.Conv2d(n_filters*8, n_filters*8, 4, 2, 1),
                                     nn.Conv2d(n_filters*16 + plus1, n_noise_channels, last_kernel, 1, 0),
                                     nn.Conv2d(n_noise_channels*2,1, 1, 1, 0)])

        self.current_channel_norms = {}
        self.mse = nn.MSELoss()
        self.lambda_ = lambda_

        self.apply(weights_init)

        # logging
        self.intermediate_Ds = []
        self.log_channel_norms = log_channel_norms
        self.detailed_logging = detailed_logging
        self.channel_norms = {layer: [] for layer in range(6)}



    def forward(self, intermediate_state_dict, detach = False):
        layer_names = list(intermediate_state_dict.keys())

        # intermediate layers
        x = intermediate_state_dict[layer_names[0]].detach() if detach else intermediate_state_dict[layer_names[0]]
        for i, layer in enumerate(self.layers):
            if i>0:
                x = torch.cat([x,
                               intermediate_state_dict[layer_names[i]].detach() if detach
                                    else intermediate_state_dict[layer_names[i]]],dim=1)

            # maybe get std dev over batch
            if self.eval_std_dev and i==4:
                x = stdDev(x)

            x = layer(x)
            # don't relu the last layer
            if i<5:
                x = self.relu(x)

            if self.log_channel_norms:
                n = self.get_channel_norm(x)
                self.current_channel_norms[i] = n
                if self.detailed_logging:
                    self.channel_norms[i].append(n.mean().item())
        if self.detailed_logging:
            self.intermediate_Ds.append(x.mean().item())

        return x.view(-1,1)

    def get_channel_norm(self, x):
        return (((x**2).mean(dim=1) + 1e-8).sqrt())

    def get_channel_norm_dist_from_1(self):
        total_dist = 0
        for layer, state in self.current_channel_norms.items():
            total_dist = total_dist + self.mse(state, torch.ones_like(state))
        return total_dist

    def get_gradient_penalty(self, inference_state_dict, generator_state_dict):

        interpolated_dict = OrderedDict()
        for (_, estate), (layer,gstate) in zip(inference_state_dict.items(), generator_state_dict.items()):

            interp = alpha_spherical_interpolate(estate.detach(), gstate.detach())
            interpolated_dict[layer] = Variable(interp, requires_grad = True)

        gp = calc_gradient_penalty(self, interpolated_dict, LAMBDA=self.lambda_)

        return gp


def stdDev(x):
    r"""
    Add a standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get its mean, over all pixels and all channels
        3) expand the layer and concatenate it with the input
    Args:
        - x (tensor): previous layer
    """
    size = x.size()
    y = torch.var(x, 0)
    y = torch.sqrt(y + 1e-8)
    y = y.view(-1)
    y = torch.mean(y)
    y = torch.ones(size[0],1,size[2],size[3]).to(x.device) * y

    return torch.cat([x, y], dim=1)



def calc_gradient_penalty(netD, interpolated_dict, LAMBDA=.1):
    """A general utility function modified from a WGAN-GP implementation.

    Not pretty rn; TODO make prettier"""

    d = interpolated_dict["Input"].device
    bs = interpolated_dict["Input"].size(0)

    disc_interpolates = netD(interpolated_dict)

    gradients = grad(outputs=disc_interpolates,
                                  inputs=[t for (_,t) in interpolated_dict.items()],
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(d),
                                  create_graph=True, retain_graph=True, only_inputs=True)

    gradient_norms = [(g.view(bs, -1).norm(2, dim=1) - 1) ** 2 for g in gradients]

    gradient_penalty = sum(gradient_norms).mean() * LAMBDA

    return gradient_penalty


class HistoricalAverageDiscriminatorLoss(nn.Module):
    r"""Historical Average Discriminator Loss from
    `"Improved Techniques for Training GANs
    by Salimans et. al." <https://arxiv.org/pdf/1606.03498.pdf>`_ paper

    Adapted from the torchgan implementation

    The loss can be described as

    .. math:: || \vtheta - \frac{1}{t} \sum_{i=1}^t \vtheta[i] ||^2

    where

    - :math:`G` : Discriminator
    - :math: `\vtheta[i]` : Discriminator Parameters at Past Timestep :math: `i`

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
        lambd (float, optional): Hyperparameter lambda for scaling the Historical Average Penalty
    """

    def __init__(self, lambd=1.0 ):
        super(HistoricalAverageDiscriminatorLoss, self).__init__()
        self.timesteps = 0
        self.sum_parameters = []
        self.lambd = lambd

    def forward(self, discriminator):
        if self.timesteps == 0:
            for p in discriminator.parameters():
                param = p.data.clone()
                self.sum_parameters.append(param)
            self.timesteps += 1
            return 0.0
        else:
            loss = 0.0
            for i, p in enumerate(discriminator.parameters()):
                loss += torch.sum(
                    (p - (self.sum_parameters[i].data / self.timesteps)) ** 2
                )
                self.sum_parameters[i] += p.data.clone()
            self.timesteps += 1
            loss *= self.lambd
            return loss

def alpha_spherical_interpolate(tensor1, tensor2):
    "Returns a tensor interpolated between these two tensors, with some random about per example in the batch."
    size = tensor1.size()
    alpha = torch.rand(size[0], 1).to(tensor1.device)

    #make 2d for interpolation
    tensor1 = tensor1.view(size[0],-1)
    tensor2 = tensor2.view(size[0],-1)

    interpolated = slerp(alpha, tensor2, tensor1)

    return interpolated.view(size)

def batchdot(A, B):
    return (A*B).sum(-1)

def slerp(interp, low, high):
    """Code lifted, torched, and batched from https://github.com/soumith/dcgan.torch/issues/14"""

    eps = 1e-12
    omega = torch.clamp(batchdot(low/(eps + torch.norm(low, dim=1).view(-1,1)),
                                 high/(eps + torch.norm(high, dim=1).view(-1,1))), -1, 1)

    omega = torch.acos(omega).view(-1, 1)
    so = torch.sin(omega)
    out = torch.where(so == 0,
                     (1.0-interp) * low + interp * high, # LERP
                     torch.sin((1.0-interp)*omega) / so * low + torch.sin(interp*omega) / so * high) #SLERP
    return out


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

def weights_init_he(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 1)
            if m.bias is not None:
                m.bias.data.zero_()

class AddNoise(nn.Module):
    """
    Adds some noise with a certain variance. During evaluation no noise is applied.

    noise_type: What sort of noise is applied after each convolutional layer? Gaussian, but of what variance?
            'fixed' = Noise is always Gaussian with variance 0.01
            'none' = No noise
            'learned_by_layer' = The variance is learned and different for each layer
            'learned_by_channel' = The variance is learned and different for each channel and each layer.
                                        Requires n_channels be set.
            'learned_filter' = Variance that is the result of a learned filter
                                        on the previous layer. Like the `reparameterization trick` of
                                        variational autoencoders.
            'poisson' = variance is equal to value, divided by 10

    Note: if `learned_filter` is used, the inputs of the previous layer are interpreted so that the first half
    of channels are the mean and the second half of channels of the variance of the distribution that is the
    outputs of the layer. In this case this module results in outputs that are not of the same shape
    as the inputs but rather of half the number of channel dimensions.

    For alo other modes, the output is of the same shape as the inputs.

    """

    def __init__(self, noise_type, n_channels = None, fixed_variance = 0.01):
        super(AddNoise, self).__init__()

        self.noise_type = noise_type
        if self.noise_type == 'fixed':
            self.fixed_variance = fixed_variance
        elif self.noise_type == 'learned_by_layer':
            self.log_sigma = nn.Parameter(torch.ones(1) * -2)
        elif self.noise_type == 'learned_by_channel':
            self.log_sigma = nn.Parameter(torch.ones(n_channels) * -2)
        elif self.noise_type == 'poisson':
            self.relu = nn.ReLU()

    def forward(self, x):

        if self.training:
            if self.noise_type == 'none':
                out = x
            elif self.noise_type == 'fixed':
                noise = torch.empty_like(x).normal_()
                out = x + noise * self.fixed_variance

            elif self.noise_type == 'learned_by_layer':
                noise = torch.empty_like(x).normal_()
                out = x + noise * torch.exp(self.log_sigma)

            elif self.noise_type == 'learned_by_channel':
                noise = torch.empty_like(x).normal_()
                out = x + noise * torch.exp(self.log_sigma)[None,:,None,None]

            elif self.noise_type == 'learned_filter':
                n_channels = x.size()[1]
                assert n_channels % 2 == 0

                mu = x[:, :n_channels//2,:,:]
                log_sigma = x[:, n_channels//2:,:,:]

                #rescale log_sigma and shrink. This is just to make the initialization the right scale
                log_sigma = log_sigma * .01 - 2

                noise = torch.empty_like(mu).normal_()
                out = mu + noise * torch.exp(log_sigma)
            elif self.noise_type == 'poisson':
                noise = torch.empty_like(x).normal_()
                out = x + noise * self.relu(x) / 10
            else:
                raise AssertionError("noise_type not in "
                                     "['none', 'fixed', 'learned_by_layer', 'learned_by_channel', "
                                     "'poisson', 'learned_filter']")
        elif self.noise_type == 'learned_filter':
            n_channels = x.size()[1]
            assert n_channels % 2 == 0

            out = x[:, :n_channels // 2, :, :]
        else:
            out = x
            
        return out


class MaybeHeRescale(nn.Module):
    """
    First rescale the outputs of the previous layer based on that layer's He constant. Then,
    add zero-mean Gaussian noise of a given variance if noise_sigma is greater than 0; else do nothing."""
    def __init__(self, in_channels, kernel, rescale = True):
        super(MaybeHeRescale, self).__init__()
        self.weight = math.sqrt(2/(in_channels * kernel**2)) if rescale else 1

    def forward(self, x):
        x = x*self.weight
        return x

def getLayerNormalizationFactor(module):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = module.weight.size()
    fan_in = prod(size[1:])

    return math.sqrt(2.0 / fan_in)

class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())