import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions import Laplace, Normal
from torch.autograd import Variable, grad
import math

class Generator(nn.Module):
    def __init__(self, noise_dim, n_filters, n_img_channels, noise_sigma = 0, backprop_to_start = True,
                 image_size = 64, batchnorm = False):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.n_filters = n_filters
        self.n_img_channels = n_img_channels
        self.backprop_to_start = backprop_to_start

        self.generative_5to4_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     noise_dim, n_filters * 8, image_size//16, 1, 0 ),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(n_filters * 8) if batchnorm else null(),
            nn.ReLU())

        self.generative_4to3_conv = nn.Sequential(
            # state size. (n_filters*8) x 4 x 4
            nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 4, 2, 1 ),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(n_filters * 4) if batchnorm else null(),
            nn.ReLU())

        self.generative_3to2_conv = nn.Sequential(
            # state size. (n_filters*4) x 8 x 8
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1 ),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(n_filters * 2) if batchnorm else null(),
            nn.ReLU())

        self.generative_2to1_conv = nn.Sequential(
            # state size. (n_filters*2) x 16 x 16
            nn.ConvTranspose2d(n_filters * 2,     n_filters, 4, 2, 1 ),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(n_filters) if batchnorm else null(),
            nn.ReLU())
            # state size. (n_filters) x 32 x 32

        self.generative_1to0_conv = nn.Sequential(
            nn.ConvTranspose2d(    n_filters,      n_img_channels, 4, 2, 1 ),
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
    def __init__(self, noise_dim, n_filters, n_img_channels, noise_sigma = 0, backprop_to_start = True,
                 image_size = 64, batchnorm = False):
        super(Inference, self).__init__()
        self.noise_dim = noise_dim
        self.n_filters = n_filters
        self.n_img_channels = n_img_channels
        self.backprop_to_start = backprop_to_start


        self.inference_4to5_conv = nn.Sequential(
            nn.Conv2d(n_filters * 8, noise_dim, image_size // 16, 1, 0 ),
            AddNoise(noise_sigma),
        )

        self.inference_3to4_conv = nn.Sequential(
            nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1 ),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(n_filters * 8) if batchnorm else null(),
            nn.ReLU())

        self.inference_2to3_conv = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1 ),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(n_filters * 4) if batchnorm else null(),
            nn.ReLU())

        self.inference_1to2_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1 ),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(n_filters * 2) if batchnorm else null(),
            nn.ReLU())

        self.inference_0to1_conv = nn.Sequential(
            nn.Conv2d(n_img_channels, n_filters, 4, 2, 1 ),
            AddNoise(noise_sigma),
            nn.BatchNorm2d(n_filters) if batchnorm else null(),
            nn.ReLU())

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
    log_intermediate_surprisals = whether to keep an internal variable storing the layer-wise surprisals during training
    log_intermediate_reconstructions = whether to keep an internal variable storing the layer-wise reconstructions
    """

    def __init__(self, noise_dim, n_filters, n_img_channels,
                 image_size = 64,
                 surprisal_sigma=1.0,
                 noise_sigma = 0,
                 log_intermediate_surprisals=False,
                 log_intermediate_reconstructions=False,
                 log_weight_alignment=False,
                 backprop_to_start_inf=True,
                 backprop_to_start_gen=True,
                 batchnorm = False):
        super(DeterministicHelmholtz, self).__init__()

        assert image_size % 16 == 0

        self.inference = Inference(noise_dim, n_filters, n_img_channels, noise_sigma, backprop_to_start_inf, image_size,
                                   batchnorm=batchnorm)
        self.generator = Generator(noise_dim, n_filters, n_img_channels, noise_sigma, backprop_to_start_gen, image_size,
                                   batchnorm=batchnorm)

        self.inference.apply(weights_init)
        self.generator.apply(weights_init)

        self.mse = nn.MSELoss()
        self.surprisal_sigma = surprisal_sigma

        self.layer_names = list(self.inference.intermediate_state_dict.keys())
        self.which_layers = range(len(self.layer_names))
        self.noise_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        # logging
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

        return ML_loss  # / float(len(self.generator.listed_modules))

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

class DiscriminatorFactorFC(nn.Module):
    """To discriminate between two full-connected layers"""

    def __init__(self, input_size, hidden_size, with_sigmoid=False, batchnorm = True, eval_std_dev = False):
        super(DiscriminatorFactorFC, self).__init__()
        self.eval_std_dev = eval_std_dev
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        d= 1 if eval_std_dev else 0
        self.fc2 = nn.Linear(hidden_size+d, 1)
        self.bn = nn.BatchNorm1d(hidden_size) if batchnorm else null()
        self.out_nonlinearity = nn.Sigmoid() if with_sigmoid else null()

    def forward(self, h1, h2):
        # reshape
        h2 = h2.view(h2.size()[0], -1)
        h1 = h1.view(h1.size()[0], -1)

        x = torch.cat([h1, h2], dim=1)

        out = self.fc1(x)
        out = self.bn(out)
        out = self.relu(out)

        if self.eval_std_dev:
            sd = torch.mean(torch.sqrt(torch.var(out,0)+1e-8))
            to_cat = torch.ones(out.size()[0],1).to(out.device) * sd
            out = torch.cat([out, to_cat], dim=1)

        out = self.fc2(out)
        out = self.out_nonlinearity(out)

        return out


class DiscriminatorFactorConv(nn.Module):
    """To discriminate between two layers separated by a convolutional operation.

    This module programatically generates a multilayer convolutional network with N layers. Since each layer
    halves the x and y dimension, the number of layers N is log2( dim_x_y ). For this reason
    the dimension must be supplied when constructing.



    INPUTS
    [conv_kernel, conv_stride, conv_pad] = the parameters of the Conv2d in the inference operation between
    the two layers (or equivalently the ConvTranspose2d of the generator)

    inner_channels_per_layer: how many hidden dimension channels (doubles as you go up the disc).
    dim_x_y: Size of the square array in the *bottom* layer. Must be divisible by 4.

    with_sigmoid = Boolean. Whether to apply a sigmoid to the output of each inner Discriminator, as required when
                        averaging (ensembling) multiple discriminators in a standard GAN with BCELoss
    avg_after_n_layers = Int > 1 (or False): convolve this many layers and, if this many layers doesn't shrink the
                            x and y dimensions to 1, just average over the x and y channels instead of continuing
                            to convolve down. Designed to prevent lower-layer discriminators from learning to
                            discriminate global image structure. Not used if set to False.
    eval_std_dev: Boolean. Calculate the std deviation of the 2nd-to-last layer(the pre-logit)  in this disciminator,
                            and use that for the decision.

    """

    def __init__(self, h1_channels, h2_channels,
                 conv_kernel, conv_stride, conv_pad, inner_channels_per_layer,
                 dim_x_y,
                 avg_after_n_layers = False, eval_std_dev = False,
                 with_sigmoid=False,
                 batchnorm = True):
        super(DiscriminatorFactorConv, self).__init__()
        assert dim_x_y % 4 == 0
        n_inter_layers = 0 if (dim_x_y<8) else int(math.log2(dim_x_y//8))

        # handle the averaging operation
        self.avg_after_n_layers = avg_after_n_layers
        if avg_after_n_layers is not False:
            assert avg_after_n_layers > 1
            n_inter_layers = min(n_inter_layers, avg_after_n_layers-2)

        ## Layer 1 combines the two layers of the inference/generator network into 1
        self.conv_over_h1 = nn.Conv2d(h1_channels, inner_channels_per_layer // 2,
                                      conv_kernel, conv_stride, conv_pad  )

            # the result of the upper conv must have the same 2d size as the lower conv
            # (even though the original layer sizes will be different)
        self.conv_over_h2 = nn.Conv2d(h2_channels, inner_channels_per_layer // 2,
                                      conv_kernel-1, 1, int((conv_kernel) / 4)  )
        self.relu = nn.LeakyReLU(.2, inplace = True)

        ## Then we pare down this combined hidden state to a logit
        # The last layer is always kernel=4, stride = 0
        # we intersperse a number of dimesion-halving layers in between depending on the original size
        self.mid_disc_layers = nn.ModuleList([nn.Sequential(nn.Conv2d((2 ** i) * inner_channels_per_layer,
                                                        (2 ** (i+1)) * inner_channels_per_layer,
                                                        4,2,1 ),
                                      nn.BatchNorm2d(inner_channels_per_layer * (2 ** (i+1))) if batchnorm else null(),
                                      nn.LeakyReLU(.2, inplace = True))
                                  for i in range(n_inter_layers)])

        n_final_dims = (2 ** n_inter_layers) * inner_channels_per_layer
        if eval_std_dev:
            n_final_dims += 1
        self.eval_std_dev = eval_std_dev

        self.last_disc_layer = nn.Conv2d(n_final_dims,
                                            1,
                                            dim_x_y//2 if dim_x_y<8 else 4, 1, 0)


        self.out_nonlinearity = nn.Sigmoid() if with_sigmoid else null()

    def forward(self, h1, h2):
        #layer 1
        conved_h1 = self.conv_over_h1(h1)
        conved_h2 = self.conv_over_h2(h2)
        x = torch.cat([conved_h1, conved_h2], dim=1)
        x = self.relu(x)

        # intermediate layers
        for layer in self.mid_disc_layers:
            x = layer(x)

        # maybe get std dev over batch
        if self.eval_std_dev:
            x = stdDev(x)

        #last layer
        x = self.last_disc_layer(x)

        # finally, maybe, apply a sigmoid nonlinearity
        x = self.out_nonlinearity(x)

        if self.avg_after_n_layers:
            x = x.mean(dim=3).mean(dim=2)

        return x

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


class Discriminator(nn.Module):
    """To discriminate between the full network state.

    Note: when calling, it takes a full dictionary of states.

    Inputs: full_architecture: a list of sizes [Input, hidden1, hidden2, z_dim]
            layer_names: a list of the names of the layers in the state_dict
            hidden_layer_size: the size of the hidden layer each discriminator
            lambda_: when using WGAN-GP, the size of the GP
            loss_type: a string. If `BCE`, then we apply a sigmoid to each sub-discriminator before averaging."""

    def __init__(self, image_size, hidden_layer_size, layer_names,
                 noise_dim, n_filters, n_img_channels,
                 lambda_=0, loss_type='wasserstein',
                 log_intermediate_Ds=False,
                 eval_std_dev=False,
                 avg_after_n_layers=False):
        super(Discriminator, self).__init__()

        self.layer_names = layer_names
        self.lambda_ = lambda_


        with_sigmoid = loss_type == 'BCE'
        batchnorm = not loss_type == "wasserstein"

        kw_args = {"inner_channels_per_layer": hidden_layer_size,
                   "with_sigmoid": with_sigmoid,
                   "batchnorm": batchnorm,
                   "eval_std_dev": eval_std_dev,
                   "avg_after_n_layers": avg_after_n_layers}

        self.discriminator_0and1 = DiscriminatorFactorConv(n_img_channels, n_filters,
                                                           4, 2, 1,
                                                           dim_x_y=image_size,
                                                           **kw_args)
        self.discriminator_1and2 = DiscriminatorFactorConv(n_filters, n_filters * 2,
                                                           4, 2, 1,
                                                           dim_x_y=image_size//2,
                                                           **kw_args)
        self.discriminator_2and3 = DiscriminatorFactorConv(n_filters * 2, n_filters * 4,
                                                           4, 2, 1,
                                                           dim_x_y=image_size//4,
                                                           **kw_args)
        self.discriminator_3and4 = DiscriminatorFactorConv(n_filters * 4, n_filters * 8,
                                                           4, 2, 1,
                                                           dim_x_y=image_size//8,
                                                           **kw_args)
        self.discriminator_4and5 = DiscriminatorFactorFC((image_size //16)** 2 * n_filters * 8 +
                                                          noise_dim,
                                                          noise_dim*2,
                                                          with_sigmoid,
                                                          batchnorm=batchnorm,
                                                          eval_std_dev = eval_std_dev)

        self.Ds = [self.discriminator_0and1, self.discriminator_1and2,
                   self.discriminator_2and3, self.discriminator_3and4, self.discriminator_4and5]

        self.which_layers = 'all'

        # init
        self.apply(weights_init)

        # logging
        self.log_intermediate_Ds = log_intermediate_Ds
        self.intermediate_Ds = {layer: [] for layer in self.layer_names}

    def forward(self, network_state_dict):
        """

        """
        if self.which_layers == 'all':
            self.which_layers = range(len(self.Ds))

        d = 0
        for i, D in enumerate(self.Ds):
            if i not in self.which_layers:
                continue

            h1 = network_state_dict[self.layer_names[i]]
            h2 = network_state_dict[self.layer_names[i + 1]]

            this_d = D(h1, h2).view(-1, 1)
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
            h2i = inference_state_dict[self.layer_names[i + 1]].detach()

            h1g = generator_state_dict[self.layer_names[i]].detach()
            h2g = generator_state_dict[self.layer_names[i + 1]].detach()

            gp = gp + calc_gradient_penalty(D, (h1i, h2i), (h1g, h2g), LAMBDA=self.lambda_)

        return gp / float(len(self.Ds))


def calc_gradient_penalty(netD, real_data_tuple, fake_data_tuple, LAMBDA=.1):
    """A general utility function modified from a WGAN-GP implementation.

    Not pretty rn; TODO make prettier"""

    batch_size = real_data_tuple[0].size()[0]

    interpolates0 = alpha_spherical_interpolate(real_data_tuple[0], fake_data_tuple[0])
    interpolates0 = Variable(interpolates0, requires_grad=True)

    interpolates1 = alpha_spherical_interpolate(real_data_tuple[1], fake_data_tuple[1])
    interpolates1 = Variable(interpolates1, requires_grad=True)

    disc_interpolates = netD(interpolates0, interpolates1)

    gradients0, gradients1 = grad(outputs=disc_interpolates, inputs=[interpolates0, interpolates1],
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(real_data_tuple[0].device),
                                  create_graph=True, retain_graph=True, only_inputs=True)

    gradients0, gradients1 = gradients0.view(batch_size, -1), gradients1.view(batch_size, -1)

    gradient_penalty = ((gradients0.norm(2, dim=1) - 1) ** 2 +
                        (gradients1.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
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