import torch
import torch.nn as nn
from torch.autograd import grad, Variable
from torch.distributions import Laplace, Normal

from collections import OrderedDict


class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, noise_dim=100, image_dim=1, image_size=32, noise_sigma = 0,backprop_to_start = True):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.image_size = image_size
        backprop_to_start = True
        self.backprop_to_start = backprop_to_start

        self.generative_4to3 = nn.Sequential(
            nn.Linear(self.noise_dim, 1024),
            AddNoise(noise_sigma),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.generative_3to2 = nn.Sequential(
            nn.Linear(1024, 128 * (self.image_size // 4) * (self.image_size // 4)),
            AddNoise(noise_sigma),
            nn.BatchNorm1d(128 * (self.image_size // 4) * (self.image_size // 4)),
            nn.ReLU(),
        )

        self.generative_2to1_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.generative_1to0_conv = nn.Sequential(
            nn.ConvTranspose2d(64, self.image_dim, 4,2,1),
            nn.Tanh(),
        )
        initialize_weights(self)

        self.intermediate_state_dict = {'Input': None,
                           'Layer1': None,
                           'Layer2': None,
                           'Layer3': None,
                           'Layer4': None}

        # list modules bottom to top. Probably a more general way exists
        self.listed_modules = [self.generative_1to0_conv,
                              self.generative_2to1_conv,
                              self.generative_3to2,
                              self.generative_4to3]

        self.intermediate_state_dict = OrderedDict([('Input', None),
                                                    ('Layer1', None),
                                                    ('Layer2', None),
                                                    ('Layer3', None),
                                                    ('Layer4', None)])
        self.layer_names = list(self.intermediate_state_dict.keys())

    def get_detached_state_dict(self):
        detached_dict = {k: None if (v is None) else v.detach() for k, v in self.intermediate_state_dict.items()}
        return detached_dict


    def forward(self, x, from_layer = 5):
        # iterate through layers and pass the noise downwards
        for i, (G, layer_name) in enumerate(zip(self.listed_modules[::-1],
                                                self.layer_names[:0:-1])):
            # Skip the topmost n layers ?
            if len(self.listed_modules) - i > from_layer:
                continue
            self.intermediate_state_dict[layer_name] = x

            if layer_name == "Layer2":
                x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))

            # this setting makes all gradient flow only go one layer back
            if not self.backprop_to_start:
                x = x.detach()

            x = G(x)


        self.intermediate_state_dict["Input"] = x

        return x

class AddNoise(nn.Module):
    """Adds zero-mean Gaussian noise of a given variance if noise_sigma is greater than 0; else do nothing."""
    def __init__(self, noise_sigma = 0):
        super(AddNoise, self).__init__()

        if noise_sigma > 0:
            self.noise_dist = Normal(torch.tensor([0.0]), torch.tensor([noise_sigma]))
        else:
            self.noise_dist = None

    def forward(self, x):
        if self.noise_dist is not None:
            noise = self.noise_dist.sample(x.size()).to(x.device).squeeze(dim=-1)
            x += noise
        return x

class Inference(nn.Module):
    """ Inverse architecture of the generative model"""

    def __init__(self, noise_dim=100, image_dim=1, image_size=32, noise_sigma = 0, backprop_to_start = True):
        super(Inference, self).__init__()
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.image_size = image_size
        backprop_to_start = True
        self.backprop_to_start = backprop_to_start

        self.inference_3to4 = nn.Sequential(
            nn.Linear(1024, self.noise_dim),
            AddNoise(noise_sigma)
        )

        self.inference_2to3 = nn.Sequential(
            nn.Linear(128 * (self.image_size // 4) * (self.image_size // 4), 1024),
            nn.BatchNorm1d(1024),
            AddNoise(noise_sigma),
            nn.ReLU(),
        )

        self.inference_1to2_conv = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.inference_0to1_conv = nn.Sequential(
            nn.Conv2d(self.image_dim, 64, 4,2,1),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        initialize_weights(self)

        # list modules bottom to top. Probably a more general way exists
        self.listed_modules = [self.inference_0to1_conv,
                               self.inference_1to2_conv,
                               self.inference_2to3,
                               self.inference_3to4]

        self.intermediate_state_dict = OrderedDict([('Input', None),
                                                    ('Layer1', None),
                                                    ('Layer2', None),
                                                    ('Layer3', None),
                                                    ('Layer4', None)])
        self.layer_names = list(self.intermediate_state_dict.keys())

    def get_detached_state_dict(self):
        detached_dict = {k: None if (v is None) else v.detach() for k, v in self.intermediate_state_dict.items()}
        return detached_dict

    def forward(self, x):
        self.intermediate_state_dict['Input'] = x
        # iterate through layers and pass the input upwards
        for F, layer_name in zip(self.listed_modules,
                                 self.layer_names[1:]):
            # this setting makes all gradient flow only go one layer back
            if not self.backprop_to_start:
                x = x.detach()

            x = F(x)

            if layer_name == "Layer2":
                x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))

            self.intermediate_state_dict[layer_name] = x

        return x


class DeterministicHelmholtz(nn.Module):
    """

    input_size = size of the input images. Result of images.size()
    noise_dim = dimension of the noise vector at the highest level
    surprisal_sigma = how strongly to follow the gradient of layer-wise surprisal
                        (or, variance of the gaussian distribution asserted when comparing predicted
                            vs. actual lower-layer activity)
    log_intermediate_surprisals = whether to keep an internal variable storing the layer-wise surprisals during training
    log_intermediate_reconstructions = whether to keep an internal variable storing the layer-wise reconstructions
    """

    def __init__(self, image_size, noise_dim, surprisal_sigma=1.0,
                 log_intermediate_surprisals=False,
                 log_intermediate_reconstructions=False,
                 log_weight_alignment=False,
                 noise_sigma = 0,
                 backprop_to_start_inf=True,
                 backprop_to_start_gen=True):
        super(DeterministicHelmholtz, self).__init__()

        if len(image_size) == 4:
            image_dim = image_size[1]
            image_edge_size = image_size[2]
            assert image_size[2] == image_size[3]
        elif len(image_size) == 3:
            image_dim = image_size[0]
            image_edge_size = image_size[1]
            assert image_size[2] == image_size[1]
        else:
            raise NotImplementedError("Image sizes are wrong.")


        self.inference = Inference(noise_dim, image_dim, image_edge_size, noise_sigma, backprop_to_start_inf)
        self.generator = Generator(noise_dim, image_dim, image_edge_size, noise_sigma, backprop_to_start_gen)

        self.generator_modules = self.generator.listed_modules
        self.inference_modules = self.inference.listed_modules

        self.noise_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.mse = nn.MSELoss()

        self.surprisal_sigma = surprisal_sigma
        self.image_size = image_edge_size

        self.layer_names = list(self.inference.intermediate_state_dict.keys())

        # This could be (manually) set to log reconstructions etc. only in certain layers
        self.which_layers = range(len(self.inference.intermediate_state_dict))

        # loggers
        self.log_intermediate_surprisals = log_intermediate_surprisals
        self.intermediate_surprisals = {layer: [] for layer in self.layer_names}

        self.log_intermediate_reconstructions = log_intermediate_reconstructions
        self.intermediate_reconstructions = {layer: [] for layer in self.layer_names}

        self.log_weight_alignment_ = log_weight_alignment
        self.weight_alignment = {layer: [] for layer in self.layer_names}

    def infer(self, x):
        return self.inference(x)

    def generate(self, x):
        return self.generator(x)

    def noise_and_generate(self, noise_layer = 4):
        """Sample from a laplacian at a given layer and propagate to the bottom.
        Must supply the state size so we can get the batch size

        Noise_layer = int, with 0 being input and 4 being the very top
        """

        if self.inference.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        noise_layer_str = list(self.inference.intermediate_state_dict.keys())[noise_layer]

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

        for i, G in enumerate(self.generator_modules):
            if i not in self.which_layers and not self.log_intermediate_surprisals:
                continue

            lower_h = self.inference.intermediate_state_dict[self.layer_names[i]].detach()
            upper_h = self.inference.intermediate_state_dict[self.layer_names[i + 1]].detach()

            # layer2 is stored unraveled, so to pass through
            # to layer1 we need to reshape it
            if i == 1:
                upper_h = upper_h.view(-1, 128, self.image_size // 4, self.image_size // 4)

            layerwise_surprisal = self.mse(G(upper_h), lower_h)
            if i in self.which_layers:
                ML_loss = ML_loss + layerwise_surprisal

            if self.log_intermediate_surprisals:
                self.intermediate_surprisals[self.layer_names[i]].append(layerwise_surprisal.item())

        ML_loss = ML_loss / self.surprisal_sigma

        return ML_loss  # / float(len(self.generator_modules))

    def log_layerwise_reconstructions(self):
        """From the current *inferential* distribution, determine (and record?) the
        error upon reconstruction at each layer (i.e. generating down and inferring back up).

        For debugging purposes only; we never backprop through this."""

        if self.inference.intermediate_state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        if self.log_intermediate_reconstructions:
            for i, (G, F) in enumerate(zip(self.generator_modules, self.inference_modules)):
                if i not in self.which_layers:
                    self.intermediate_reconstructions[self.layer_names[i]].append(-1)
                    continue

                upper_h = self.inference.intermediate_state_dict[self.layer_names[i + 1]].detach()
                # layer2 is stored unraveled, so to pass through
                # to layer1 we need to reshape it
                if i == 1:
                    upper_h = upper_h.view(-1, 128, self.image_size // 4, self.image_size // 4)

                generated_lower_h = G(upper_h)
                reconstructed_upper_h = F(generated_lower_h)

                error = self.mse(upper_h, reconstructed_upper_h)
                self.intermediate_reconstructions[self.layer_names[i]].append(error.item())

    def log_weight_alignment(self):
        if self.log_weight_alignment_:
            for i, (G, F) in enumerate(zip(self.generator_modules, self.inference_modules)):
                gen_weight = list(G.parameters())[0]
                inf_weight = list(F.parameters())[0]

                cosine = torch.nn.CosineSimilarity()(inf_weight.cpu().view(1, -1),
                                                     gen_weight.cpu().view(1, -1))
                angle = torch.acos(cosine).item()
                self.weight_alignment[self.layer_names[i]].append(angle)

def null(x):
    "Pickleable nothing"
    return x

class DiscriminatorFactor(nn.Module):
    """To discriminate between two layers"""

    def __init__(self, input_size, with_sigmoid=False):
        super(DiscriminatorFactor, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)

        self.out_nonlinearity = nn.Sigmoid() if with_sigmoid else null

    def forward(self, x):

        out = self.fc1(x)
        out = self.out_nonlinearity(out)

        return out


class DiscriminatorFactorConv(nn.Module):
    """To discriminate between activations on a convolutional layer

    with_sigmoid = Boolean. Whether to apply a sigmoid to the output of each inner Discriminator, as required when
                        averaging (ensembling) multiple discriminators in a standard GAN with BCELoss"""

    def __init__(self, h1_channels,
                 conv_kernel, conv_stride, conv_pad,
                 with_sigmoid=False):
        super(DiscriminatorFactorConv, self).__init__()
        self.conv_over_h1 = nn.Conv2d(h1_channels, 1,
                                      conv_kernel, conv_stride, conv_pad,
                                      bias = False)

        self.out_nonlinearity = nn.Sigmoid() if with_sigmoid else null

    def forward(self, h1):
        x = self.conv_over_h1(h1)
        x = x.view(h1.size()[0],-1)
        # finally, maybe, apply a sigmoid nonlinearity
        x = self.out_nonlinearity(x)

        return x.mean(dim=1)


class Discriminator(nn.Module):
    """To discriminate between the full network state.

    Note: when calling, it takes a full dictionary of states.

    Inputs: full_architecture: a list of sizes [Input, hidden1, hidden2, z_dim]
            layer_names: a list of the names of the layers in the state_dict
            hidden_layer_size: the size of the hidden layer each discriminator
            lambda_: when using WGAN-GP, the size of the GP
            loss_type: a string. If `BCE`, then we apply a sigmoid to each sub-discriminator before averaging."""

    def __init__(self, layer_names, lambda_=0, loss_type='wasserstein',
                 noise_dim=100,
                 log_intermediate_Ds=False):
        super(Discriminator, self).__init__()

        self.layer_names = layer_names
        self.lambda_ = lambda_

        with_sigmoid = loss_type == 'BCE'

        self.discriminator_0 = DiscriminatorFactorConv(1,
                                                       4,2,1,
                                                       with_sigmoid = with_sigmoid)
        self.discriminator_1= DiscriminatorFactorConv(64,
                                                       4, 2, 1,
                                                       with_sigmoid = with_sigmoid)
        self.discriminator_2= DiscriminatorFactor(128*7*7,
                                                       with_sigmoid = with_sigmoid)
        self.discriminator_3 = DiscriminatorFactor(1024,
                                                         with_sigmoid)
        self.discriminator_4 = DiscriminatorFactor(noise_dim,
                                                         with_sigmoid)

        self.Ds = [self.discriminator_0, self.discriminator_1,
                   self.discriminator_2, self.discriminator_3, self.discriminator_4]

        self.log_intermediate_Ds = log_intermediate_Ds
        self.intermediate_Ds = {layer: [] for layer in self.layer_names}

        self.which_layers = 'all'

    def forward(self, network_state_dict):
        """
        A note on inference_or_generation:
        If the backprop_through_full_FG flag is False, then we need to detach part of the
        network state before feeding it to the (local) discriminator. Please, when calling the discriminator,
        tell it which you would like to detach by telling it if inference or generation is happening.
        """
        if self.which_layers == 'all':
            self.which_layers = range(len(self.Ds))

        d = 0
        for i, D in enumerate(self.Ds):
            if i not in self.which_layers:
                continue

            h1 = network_state_dict[self.layer_names[i]]

            this_d = D(h1).view(-1, 1)

            d = d + this_d

            if self.log_intermediate_Ds:
                self.intermediate_Ds[self.layer_names[i]].append(this_d.mean().item())

        return d / float(len(self.Ds))

    def get_gradient_penalty(self, inference_state_dict, generator_state_dict):

        gp = 0
        for i, D in enumerate(self.Ds):
            if i not in self.which_layers:
                continue

            h1i = inference_state_dict[self.layer_names[i]].detach()

            h1g = generator_state_dict[self.layer_names[i]].detach()

            gp = gp + calc_gradient_penalty(D, h1i, h1g, LAMBDA=self.lambda_)

        return gp / float(i)

def alpha_spherical_interpolate(tensor1, tensor2):
    "Returns a tensor interpolated between these two tensors, with some random alpha per example in the batch."
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


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA=.1):
    """A general utility function modified from a WGAN-GP implementation.
    """

    batch_size = real_data.size()[0]

    interpolated = alpha_spherical_interpolate(real_data, fake_data)
    interpolated = Variable(interpolated, requires_grad=True)

    disc_interpolates = netD(interpolated)

    gradients = grad(outputs=disc_interpolates, inputs=interpolated,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients= gradients.view(batch_size, -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty



def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()