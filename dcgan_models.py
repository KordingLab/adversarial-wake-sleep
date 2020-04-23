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
                 image_size = 64, batchnorm = False, selu = False, dropout= False, stochastic_binary = False):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.n_filters = n_filters
        self.n_img_channels = n_img_channels
        self.backprop_to_start = backprop_to_start
        self.stochastic_binary = stochastic_binary

        # A small note regarding he initialization (if used) for those curious:
        # An interesting thing with the ConvTranspose is that the number of effective inputs is not
        # kernel_size ** 2 * in_channels, but rather depends on the stride and size (due to expansion
        # of the inputs and convolving with padding; both implicitly apply 0s)

        # In the learned_filter mode, all conv layers need to output twice the number of channels as before.
        maybetimestwo = 2 if noise_type=="learned_filter" else 1

        if stochastic_binary:
            self.generative_5to4_conv = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(noise_dim, n_filters * 8 * maybetimestwo, image_size // 16, 1, 0),
                nn.Sigmoid())

            self.generative_4to3_conv = nn.Sequential(
                # state size. (n_filters*8) x 4 x 4
                nn.ConvTranspose2d(n_filters * 8, n_filters * 4 * maybetimestwo, 4, 2, 1),
                nn.Sigmoid())

            self.generative_3to2_conv = nn.Sequential(
                # state size. (n_filters*4) x 8 x 8
                nn.ConvTranspose2d(n_filters * 4, n_filters * 2 * maybetimestwo, 4, 2, 1),
                nn.Sigmoid())


            self.generative_2to1_conv = nn.Sequential(
                # state size. (n_filters*2) x 16 x 16
                nn.ConvTranspose2d(n_filters * 2, n_filters * maybetimestwo, 4, 2, 1),
                nn.Sigmoid())

            # state size. (n_filters) x 32 x 32

            self.generative_1to0_conv = nn.Sequential(
                nn.ConvTranspose2d(n_filters, n_img_channels * maybetimestwo, 4, 2, 1),
                nn.Sigmoid())

        else:
            self.generative_5to4_conv = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(noise_dim, n_filters * 8 * maybetimestwo, image_size//16, 1, 0 ),
                AddNoise(noise_type, n_filters * 8),
                nn.BatchNorm2d(n_filters * 8) if batchnorm else null(),
                nn.SELU() if selu else nn.ReLU(),
                nn.AlphaDropout(.3) if dropout else null())

            self.generative_4to3_conv = nn.Sequential(
                # state size. (n_filters*8) x 4 x 4
                nn.ConvTranspose2d(n_filters * 8, n_filters * 4 * maybetimestwo, 4, 2, 1 ),
                AddNoise(noise_type, n_filters * 4),
                nn.BatchNorm2d(n_filters * 4) if batchnorm else null(),
                nn.SELU() if selu else nn.ReLU(),
                nn.AlphaDropout(.3) if dropout else null())

            self.generative_3to2_conv = nn.Sequential(
                # state size. (n_filters*4) x 8 x 8
                nn.ConvTranspose2d(n_filters * 4, n_filters * 2 * maybetimestwo, 4, 2, 1 ),
                AddNoise(noise_type, n_filters * 2),
                nn.BatchNorm2d(n_filters * 2) if batchnorm else null(),
                nn.SELU() if selu else  nn.ReLU(),
                nn.AlphaDropout(.3) if dropout else null())

            self.generative_2to1_conv = nn.Sequential(
                # state size. (n_filters*2) x 16 x 16
                nn.ConvTranspose2d(n_filters * 2, n_filters * maybetimestwo, 4, 2, 1 ),
                AddNoise(noise_type, n_filters),
                nn.BatchNorm2d(n_filters) if batchnorm else null(),
                nn.SELU() if selu else nn.ReLU(),
                nn.AlphaDropout(.3) if dropout else null())
                # state size. (n_filters) x 32 x 32

            self.generative_1to0_conv = nn.Sequential(
                nn.ConvTranspose2d(n_filters, n_img_channels * maybetimestwo, 4, 2, 1 ),
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

    def forward(self, x, from_layer = 5, update_states = True):

        if self.stochastic_binary and from_layer==5:
            x.bernoulli_(.4)

        # iterate through layers and pass the noise downwards
        for i, (G, layer_name) in enumerate(zip(self.listed_modules[::-1],
                                                self.layer_names[:0:-1])):
            # Skip the topmost n layers ?
            if len(self.listed_modules) - i > from_layer:
                continue
            if update_states:
                self.intermediate_state_dict[layer_name] = x

            # this setting makes all gradient flow only go one layer back
            if not self.backprop_to_start:
                x = x.detach()
            x = G(x)

            if self.stochastic_binary:
                if self.training:
                    x = torch.bernoulli(x).detach()
                else:
                    x = torch.round(x)

        if update_states:
            self.intermediate_state_dict["Input"] = x

        return x


class Inference(nn.Module):
    def __init__(self, noise_dim, n_filters, n_img_channels, noise_type = 'none', backprop_to_start = True,
                 image_size = 64, batchnorm = False, selu=False, dropout = False,
                 stochastic_binary = False, noise_before = False):
        super(Inference, self).__init__()
        self.noise_dim = noise_dim
        self.n_filters = n_filters
        self.n_img_channels = n_img_channels
        self.backprop_to_start = backprop_to_start
        self.stochastic_binary = stochastic_binary

        # In the learned_filter mode, all conv layers need to output twice the number of channels as before.
        maybetimestwo = 2 if noise_type=="learned_filter" else 1

        maybenoise = 3 if noise_before else 0

        # this would be simple to build programmatically in the future with ModuleList, but explicit for now.
        if stochastic_binary:
            self.inference_4to5_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_filters * 8 + maybenoise, noise_dim * maybetimestwo, image_size // 16, 1, 0),
                nn.Sigmoid())


            self.inference_3to4_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_filters * 4 + maybenoise, n_filters * 8 * maybetimestwo, 4, 2, 1),
                nn.Sigmoid())

            self.inference_2to3_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_filters * 2 + maybenoise, n_filters * 4 * maybetimestwo, 4, 2, 1),
                nn.Sigmoid())


            self.inference_1to2_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_filters + maybenoise, n_filters * 2 * maybetimestwo, 4, 2, 1),
                nn.Sigmoid())


            self.inference_0to1_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_img_channels + maybenoise, n_filters * maybetimestwo, 4, 2, 1),
                nn.Sigmoid())

        else:
            self.inference_4to5_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_filters * 8 + maybenoise, noise_dim * maybetimestwo, image_size // 16, 1, 0 ),
                AddNoise(noise_type, noise_dim)
            )

            self.inference_3to4_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_filters * 4 + maybenoise, n_filters * 8 * maybetimestwo, 4, 2, 1 ),
                AddNoise(noise_type, n_filters * 8),
                nn.BatchNorm2d(n_filters * 8) if batchnorm else null(),
                nn.SELU() if selu else nn.ReLU(),
                nn.AlphaDropout(.3) if dropout else null())

            self.inference_2to3_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_filters * 2 + maybenoise, n_filters * 4 * maybetimestwo, 4, 2, 1 ),
                AddNoise(noise_type, n_filters * 4),
                nn.BatchNorm2d(n_filters * 4) if batchnorm else null(),
                nn.SELU() if selu else nn.ReLU(),
                nn.AlphaDropout(.3) if dropout else null())

            self.inference_1to2_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_filters + maybenoise, n_filters * 2 * maybetimestwo, 4, 2, 1 ),
                AddNoise(noise_type, n_filters * 2),
                nn.BatchNorm2d(n_filters * 2) if batchnorm else null(),
                nn.SELU() if selu else nn.ReLU(),
                nn.AlphaDropout(.3) if dropout else null())

            self.inference_0to1_conv = nn.Sequential(
                NoiseChannel() if noise_before else null(),
                nn.Conv2d(n_img_channels + maybenoise, n_filters * maybetimestwo, 4, 2, 1 ),
                AddNoise(noise_type, n_filters),
                nn.BatchNorm2d(n_filters) if batchnorm else null(),
                nn.SELU() if selu else nn.ReLU(),
                nn.AlphaDropout(.3) if dropout else null())


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
        #binarize the input
        if self.stochastic_binary:
            # map to [0,1]
            x = torch.round(x/2+.5)

        self.intermediate_state_dict['Input'] = x
        # iterate through layers and pass the input upwards
        for F, layer_name in zip(self.listed_modules,
                                 self.layer_names[1:]):
            # this setting makes all gradient flow only go one layer back
            if not self.backprop_to_start:
                x = x.detach()

            x = F(x)
            if self.stochastic_binary:
                if self.training:
                    x = torch.bernoulli(x).detach()
                else:
                    x = torch.round(x)

            self.intermediate_state_dict[layer_name] = x

        return x


class Helmholtz(nn.Module):
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
                 stochastic_binary =False,
                 selu = False,
                 dropout = False,
                 noise_before = False):
        super(Helmholtz, self).__init__()

        assert image_size % 16 == 0

        self.inference = Inference(noise_dim, n_filters, n_img_channels, noise_type, backprop_to_start_inf, image_size,
                                   batchnorm=batchnorm,
                                   selu = selu, dropout = dropout, stochastic_binary = stochastic_binary,
                                   noise_before = noise_before)
        self.generator = Generator(noise_dim, n_filters, n_img_channels, noise_type, backprop_to_start_gen, image_size,
                                   batchnorm=batchnorm,stochastic_binary = stochastic_binary,
                                   selu = selu, dropout = dropout)

        # init

        self.inference.apply(weights_init)
        self.generator.apply(weights_init)

        self.surprisal_loss = nn.MSELoss() if stochastic_binary else nn.L1Loss()
        self.mse = nn.MSELoss()

        self.surprisal_sigma = surprisal_sigma

        self.layer_names = list(self.inference.intermediate_state_dict.keys())
        self.which_layers = range(len(self.layer_names))
        self.noise_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.noise_before = noise_before

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

            layerwise_surprisal = self.surprisal_loss(x, lower_h)

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

            layerwise_surprisal = self.surprisal_loss(upper_h, F_upper_h)
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


    def layerwise_reconstructions(self):
        """From the current *inferential* distribution, determine (and record?) the
        error upon reconstruction at each layer (i.e. generating down and inferring back up).

        For debugging purposes only; we never backprop through this."""

        if self.inference.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        total_loss = 0
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
            total_loss = total_loss + error
            if self.log_intermediate_reconstructions:
                self.intermediate_reconstructions[self.layer_names[i]].append(error.item())
        return total_loss

    def reconstruct_back_down(self, also_inference = False):
        """From the current *inferential* distribution, pass the state back down from each layer (through
        the generative model) down to the inputs. Minimize the error."""
        if self.inference.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        l1_loss = nn.L1Loss()

        total_loss = 0
        for i,layer in enumerate(self.layer_names):
            if layer == 'Input':
                continue

            activations = self.inference.intermediate_state_dict[layer]
            if not also_inference:
                activations = activations.detach()

            gen_img = self.generator(activations,
                                     from_layer = i, update_states = False)

            total_loss = total_loss + l1_loss(self.inference.intermediate_state_dict['Input'], gen_img)

        return total_loss

    def log_weight_alignment(self):
        if self.log_weight_alignment_:
            for i, (G, F) in enumerate(zip(self.generator.listed_modules, self.inference.listed_modules)):
                gen_weight = list(G.parameters())[0]
                inf_weight = list(F.parameters())[0]
                if self.noise_before:
                    inf_weight = inf_weight[:,:-3]

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

class NoiseChannel(nn.Module):
    """Adds some additional channels that are pure uniform noise"""

    def __init__(self,n_channels = 3):
        super(NoiseChannel, self).__init__()
        self.n_channels = n_channels


    def forward(self, x):
        if self.training:
            noise = torch.empty_like(x)[:,:self.n_channels,:,:].uniform_()
        else:
            noise = torch.zeros_like(x)[:, :self.n_channels, :, :]
        x = torch.cat([x,noise],dim=1)
        return x


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
        elif self.noise_type == 'poisson' or self.noise_type == 'exponential':
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
            elif self.noise_type == 'exponential':
                # nice one-liner
                noise = torch.empty_like(x).uniform_(1e-8,1).reciprocal_().log_()
                # noise = self.relu(noise - 1)
                out = x + noise
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
