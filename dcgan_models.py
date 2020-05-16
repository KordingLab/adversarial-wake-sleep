import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable, grad
import math
from numpy import prod
from utils import stdDev, weights_init

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
    def __init__(self, noise_dim, n_filters, n_img_channels, image_size = 32,hard_norm = False):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.n_filters = n_filters
        self.n_img_channels = n_img_channels


        self.generative_5to4_conv = nn.ConvTranspose2d(noise_dim, n_filters * 8, image_size//16, 1, 0 )
        self.generative_4to3_conv = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 4, 2, 1 )
        self.generative_3to2_conv = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1 )
        self.generative_2to1_conv = nn.ConvTranspose2d(n_filters * 2, n_filters,     4, 2, 1 )
        self.generative_1to0_conv = nn.ConvTranspose2d(n_filters, n_img_channels, 4, 2, 1 )

        self.normalizer = NormalizationLayer() if hard_norm else null()

        # list modules bottom to top. Probably a more general way exists
        self.listed_modules = [self.generative_1to0_conv,
                              self.generative_2to1_conv,
                              self.generative_3to2_conv,
                              self.generative_4to3_conv,
                              self.generative_5to4_conv]

        self.activations = OrderedDict([('Layer1', nn.Tanh()),
                                        ('Layer2', nn.ReLU()),
                                        ('Layer3', nn.ReLU()),
                                        ('Layer4', nn.ReLU()),
                                        ('Layer5', nn.ReLU())])

        self.intermediate_state_dict = OrderedDict([('Input',  None),
                                                    ('Layer1', None),
                                                    ('Layer2', None),
                                                    ('Layer3', None),
                                                    ('Layer4', None),
                                                    ('Layer5', None)])
        self.layer_names = list(self.intermediate_state_dict.keys())
        weights_init(self)

        # noise applied
        self.sigma2 = 0.01

    def forward(self, x, from_layer=5, update_states=True):
        x = self.normalizer(x)

        # iterate through layers and pass the noise downwards
        for i, (G, layer_name) in enumerate(zip(self.listed_modules[::-1],
                                                self.layer_names[:0:-1])):

            if from_layer < 5 - i:
                continue

            if update_states:
                self.intermediate_state_dict[layer_name] = x

            x = G(x)

            if layer_name != "Layer1":
                x = self.normalizer(x)
                if self.training:
                    noise = torch.empty_like(x).normal_() * self.sigma2
                    x += noise
            x = self.activations[layer_name](x)

        if update_states:
            self.intermediate_state_dict["Input"] = x

        return x

class Inference(nn.Module):
    def __init__(self, n_latents, n_filters, n_img_channels, image_size = 32, bn = True,
                            noise_before=False, hard_norm=False, spec_norm=True, derelu = True):
        super(Inference, self).__init__()
        self.n_latents = n_latents
        self.n_filters = n_filters
        self.n_img_channels = n_img_channels
        p1=1 if noise_before else 0
        self.noise_before = noise_before

        self.inference_5from4_conv = BasicBlock(n_filters * 8 + p1, n_latents, image_size//16, 1, 0,
                                                spec_norm = spec_norm, bn=bn, derelu = derelu)
        self.inference_4from3_conv = BasicBlock(n_filters * 4 + p1, n_filters * 8, 4, 2, 1,
                                                spec_norm = spec_norm, bn=bn, derelu = derelu)
        self.inference_3from2_conv = BasicBlock(n_filters * 2 + p1, n_filters * 4, 4, 2, 1,
                                                spec_norm = spec_norm, bn=bn, derelu = derelu)
        self.inference_2from1_conv = BasicBlock(n_filters + p1, n_filters * 2,     4, 2, 1,
                                                spec_norm = spec_norm, bn=bn, derelu = derelu)
        self.inference_1from0_conv = BasicBlock(n_img_channels + p1, n_filters   , 4, 2, 1, derelu=False,
                                                 bn=bn, spec_norm = spec_norm )

        self.normalizer = NormalizationLayer() if hard_norm else null()

        self.listed_modules = [self.inference_1from0_conv,
                               self.inference_2from1_conv,
                               self.inference_3from2_conv,
                               self.inference_4from3_conv,
                               self.inference_5from4_conv]

        self.intermediate_state_dict = OrderedDict([('Input', None),
                                                    ('Layer1', None),
                                                    ('Layer2', None),
                                                    ('Layer3', None),
                                                    ('Layer4', None),
                                                    ('Layer5', None)])
        self.activations = OrderedDict([('Layer1', nn.ReLU()),
                                        ('Layer2', nn.ReLU()),
                                        ('Layer3', nn.ReLU()),
                                        ('Layer4', nn.ReLU()),
                                        ('Layer5', null())])
        self.layer_names = list(self.intermediate_state_dict.keys())
        weights_init(self)

        #noise applied after each conv
        self.sigma2 = 0.01

    def forward(self, x, to_layer=5, update_states=True):
        self.intermediate_state_dict['Input'] = x
        # iterate through layers and pass the input upwards
        for i, (F, layer_name) in enumerate(zip(self.listed_modules,
                                                self.layer_names[1:])):

            if i >= to_layer:
                continue
            if self.noise_before:
                if self.training:
                    noise = torch.empty(x.size(0), 1, x.size(2), x.size(3)).normal_().to(x.device)
                else:
                    noise = torch.zeros(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
                x = torch.cat([x, noise], dim=1)
            x = F(x, self.sigma2)

            x = self.activations[layer_name](x)
            x = self.normalizer(x)
            if update_states:
                self.intermediate_state_dict[layer_name] = x

        return x

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel, stride, padding, derelu=True, spec_norm = False, bn = True):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.planes = planes
        if spec_norm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(inplanes, planes, kernel, stride, padding, bias=False))
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel, stride, padding, bias=False)
        self.bn1 = norm_layer(planes) if bn else null()
        self.derelu = DeReLU(inplanes) if derelu else null()


    def forward(self, out, sigma2 = 0.01):
        out = self.derelu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        noise = torch.empty_like(out).normal_() * sigma2
        out = out + noise

        return out

class DeReLU(nn.Module):
    """The stochastic "inverse" of a ReLU.
    Puts random negative noise where there used to be zeros. """
    def __init__(self, input_channels):
        super(DeReLU, self).__init__()

        self.scale = nn.Parameter(.5*torch.ones(1,input_channels,1,1), requires_grad = False)

    def forward(self,x):
        #exponentially distributed negative noise
        noise = torch.empty_like(x).uniform_(1e-8, 1).log_() * self.scale
        # place where it's zero
        out = torch.where(x>0, x, noise)
        return out


class null(nn.Module):
    "Pickleable nothing"
    def __init__(self):
        super(null, self).__init__()

    def forward(self, x):
        return x


class Discriminator(nn.Module):
    """Takes both the input and latent state."""

    def __init__(self, n_latents, n_filters, n_img_channels, image_size=32, hidden_dim=100, hard_norm=False,
                                dropout = 0):
        super(Discriminator, self).__init__()

        self.linear01 = (nn.Linear(n_img_channels * image_size ** 2 + n_filters * (image_size//2) ** 2 + 2, hidden_dim))
        self.linear12 = (nn.Linear(n_filters * (image_size//2) ** 2 + n_filters * 2 * (image_size//4) ** 2 + 2, hidden_dim))
        self.linear23 = (nn.Linear(n_filters * 2 * (image_size//4) ** 2 + n_filters * 4 * (image_size//8) ** 2 + 2, hidden_dim))
        self.linear34 = (nn.Linear(n_filters * 4 * (image_size//8) ** 2 + n_filters * 8 * (image_size//16) ** 2 + 2, hidden_dim))
        self.linear45 = (nn.Linear(n_filters * 8 * (image_size//16) ** 2 + n_latents + 2, hidden_dim))

        self.linear2 = (nn.Linear(hidden_dim * 5, 1))
        self.relu = nn.LeakyReLU()
        self.normalizer = NormalizationLayer() if hard_norm else null()
        self.dropout = nn.Dropout(dropout)

        weights_init(self)

    def forward(self, state):
        ins = [stdDev(s.view(s.size(0), -1)) for _, s in state.items()]

        x = torch.cat([self.linear01(torch.cat([ins[0], ins[1]], dim=1)),
                       self.linear12(torch.cat([ins[1], ins[2]], dim=1)),
                       self.linear23(torch.cat([ins[2], ins[3]], dim=1)),
                       self.linear34(torch.cat([ins[3], ins[4]], dim=1)),
                       self.linear45(torch.cat([ins[4], ins[5]], dim=1))], dim=1)

        x = self.relu(x)
        x = self.dropout(x)
        x = self.normalizer(x)
        x = self.linear2(x)

        return x


class ReadoutDiscriminator(nn.Module):
    def __init__(self, n_filters, image_size,
                 spec_norm=True):
        super(ReadoutDiscriminator, self).__init__()
        # number of features in the top level (before latents)
        self.n_features = n_filters * 8 * (image_size // 16) ** 2
        if spec_norm:
            self.readout = nn.utils.spectral_norm(nn.Linear(self.n_features + 1, 1))
        else:
            self.readout = nn.Linear(self.n_features + 1, 1)

        weights_init(self)

    def forward(self, inference_layer):

        x = self.readout(stdDev(inference_layer.view(-1, self.n_features)))

        return x


class NoiseChannel(nn.Module):
    """Adds some additional channels that are pure uniform noise"""

    def __init__(self,n_channels = 3):
        super(NoiseChannel, self).__init__()
        self.n_channels = n_channels


    def forward(self, x):
        if self.training:
            noise = torch.empty_like(x)[:,:self.n_channels,:,:].uniform_(1e-8,1).reciprocal_().log_()*0.02
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

    def __init__(self, noise_type, n_channels = None, fixed_variance = 0.02):
        super(AddNoise, self).__init__()

        self.noise_type = noise_type

        self.fixed_variance = fixed_variance
        if self.noise_type == 'learned_by_layer':
            self.log_sigma = nn.Parameter(torch.ones(1) * -2)
        elif self.noise_type == 'learned_by_channel':
            self.log_sigma = nn.Parameter(torch.ones(n_channels) * -2)
        elif self.noise_type == 'poisson' or self.noise_type == 'exponential':
            self.relu = nn.ReLU()

        self.decay = 1.

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
                noise = torch.empty_like(x).uniform_(1e-8, 1).reciprocal_().log_()
                # noise = self.relu(noise - 1)
                out = x + self.decay * noise
            elif self.noise_type == 'laplace':
                noise = torch.empty_like(x).uniform_(1e-8, 1).reciprocal_().log_()
                noise *= (torch.empty_like(x).bernoulli_() * 2) - 1
                out = x + self.decay * noise * self.fixed_variance
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

            #         self.decay *= .9999
        return out



class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())