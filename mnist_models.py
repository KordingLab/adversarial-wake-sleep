import torch
import torch.nn as nn
from torch.autograd import grad, Variable
from torch.distributions import Laplace, Normal


class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, noise_dim=100, image_dim=1, image_size=32):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.image_size = image_size

        self.generative_4to3 = nn.Sequential(
            nn.Linear(self.noise_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU()
        )

        self.generative_3to2 = nn.Sequential(
            nn.Linear(1024, 128 * (self.image_size // 4) * (self.image_size // 4)),
            nn.BatchNorm1d(128 * (self.image_size // 4) * (self.image_size // 4)),
            nn.LeakyReLU(),
        )

        self.generative_2to1_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.generative_1to0_conv = nn.Sequential(
            nn.ConvTranspose2d(64, self.image_dim, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

        self.state_dict = {'Input': None,
                           'Layer1': None,
                           'Layer2': None,
                           'Layer3': None,
                           'Layer4': None}

    def get_detached_state_dict(self):
        detached_dict = {k: None if (v is None) else v.detach() for k, v in self.state_dict.items()}
        return detached_dict

    def forward(self, x):
        self.state_dict['Layer4'] = x

        x = self.generative_4to3(x)
        self.state_dict['Layer3'] = x

        x = self.generative_3to2(x)
        self.state_dict['Layer2'] = x
        # ^ layer 2 saved as FC

        x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
        x = self.generative_2to1_conv(x)
        self.state_dict['Layer1'] = x

        x = self.generative_1to0_conv(x)
        self.state_dict['Input'] = x

        return x


class Inference(nn.Module):
    """ Inverse architecture of the generative model"""

    def __init__(self, noise_dim=100, image_dim=1, image_size=32):
        super(Inference, self).__init__()
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.image_size = image_size

        self.inference_3to4 = nn.Sequential(
            nn.Linear(1024, self.noise_dim),
        )

        self.inference_2to3 = nn.Sequential(
            nn.Linear(128 * (self.image_size // 4) * (self.image_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
        )

        self.inference_1to2_conv = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.inference_0to1_conv = nn.Sequential(
            nn.Conv2d(self.image_dim, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        initialize_weights(self)

        self.state_dict = {'Input': None,
                           'Layer1': None,
                           'Layer2': None,
                           'Layer3': None,
                           'Layer4': None}

    def get_detached_state_dict(self):
        detached_dict = {k: None if (v is None) else v.detach() for k, v in self.state_dict.items()}
        return detached_dict

    def forward(self, input):
        self.state_dict['Input'] = input

        x = self.inference_0to1_conv(input)
        self.state_dict['Layer1'] = x

        x = self.inference_1to2_conv(x)
        x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
        self.state_dict['Layer2'] = x
        # ^ layer 2 saved as FC

        x = self.inference_2to3(x)
        self.state_dict['Layer3'] = x

        x = self.inference_3to4(x)
        self.state_dict['Layer4'] = x

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
                 log_weight_alignment=False):
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

        self.inference = Inference(noise_dim, image_dim, image_edge_size)
        self.generator = Generator(noise_dim, image_dim, image_edge_size)

        # list modules bottom to top. Probably a more general way
        self.generator_modules = [self.generator.generative_1to0_conv,
                                  self.generator.generative_2to1_conv,
                                  self.generator.generative_3to2,
                                  self.generator.generative_4to3]

        self.inference_modules = [self.inference.inference_0to1_conv,
                                  self.inference.inference_1to2_conv,
                                  self.inference.inference_2to3,
                                  self.inference.inference_3to4]

        self.mse = nn.MSELoss()
        self.surprisal_sigma = surprisal_sigma
        self.image_size = image_edge_size

        self.layer_names = list(self.inference.state_dict.keys())
        self.log_intermediate_surprisals = log_intermediate_surprisals
        if log_intermediate_surprisals:
            self.intermediate_surprisals = {layer: [] for layer in self.layer_names}

        self.log_intermediate_reconstructions = log_intermediate_reconstructions
        if log_intermediate_reconstructions:
            self.intermediate_reconstructions = {layer: [] for layer in self.layer_names}
        self.log_weight_alignment_ = log_weight_alignment
        if log_weight_alignment:
            self.weight_alignment = {layer: [] for layer in self.layer_names}

        self.which_layers = range(len(self.inference.state_dict))
        self.noise_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def infer(self, x):
        return self.inference(x)

    def generate(self, x):
        return self.generator(x)

    def noise_and_generate(self, noise_layer):
        """Sample from a laplacian at a given layer and propagate to the bottom.
        Must supply the state size so we can get the batch size

        Noise_layer = int, with 0 being input and 4 being the very top
        """

        if self.inference.state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        noise_layer_str = list(self.inference.state_dict.keys())[noise_layer]

        x = self.noise_dist.sample(self.inference.state_dict[noise_layer_str].size()).to(
            self.inference.state_dict[noise_layer_str].device)
        x = x.squeeze(dim=-1)
        # x = torch.abs(x)

        if noise_layer > 3:
            self.generator.state_dict['Layer4'] = x

            x = self.generator.generative_4to3(x)

        if noise_layer > 2:
            self.generator.state_dict['Layer3'] = x
            x = self.generator.generative_3to2(x)

        if noise_layer > 1:
            self.generator.state_dict['Layer2'] = x
            # ^ layer 2 saved as FC

            x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
            x = self.generator.generative_2to1_conv(x)
        if noise_layer > 0:
            self.generator.state_dict['Layer1'] = x

            x = self.generator.generative_1to0_conv(x)
            self.generator.state_dict['Input'] = x
        else:
            raise AssertionError("Noising the input layer? that doesn't make that much sense.")

        return x

    def generator_surprisal(self):
        """Given the current inference state, ascertain how surprised the generator model was.

        """
        # here we have to a assume a noise model in order to calculate p(h_1 | h_2 ; G)
        # with Gaussian we have log p  = MSE between actual and predicted

        if self.inference.state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        ML_loss = 0

        for i, G in enumerate(self.generator_modules):
            if i not in self.which_layers and not self.log_intermediate_surprisals:
                continue

            lower_h = self.inference.state_dict[self.layer_names[i]].detach()
            upper_h = self.inference.state_dict[self.layer_names[i + 1]].detach()

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

        if self.inference.state_dict['Input'] is None:
            raise AssertionError("Inference must be run first before calculating this.")

        if self.log_intermediate_reconstructions:
            for i, (G, F) in enumerate(zip(self.generator_modules, self.inference_modules)):
                if i not in self.which_layers:
                    self.intermediate_reconstructions[self.layer_names[i]].append(-1)
                    continue

                upper_h = self.inference.state_dict[self.layer_names[i + 1]].detach()
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

                cosine = torch.nn.CosineSimilarity()(inf_weight.transpose(1, 0).cpu().view(1, -1),
                                                     gen_weight.cpu().view(1, -1))
                angle = torch.acos(cosine).item()
                self.weight_alignment[self.layer_names[i]].append(angle)

def null(x):
    "Pickleable nothing"
    return x

class DiscriminatorFactorFC(nn.Module):
    """To discriminate between two full-connected layers"""

    def __init__(self, input_size, hidden_size, with_sigmoid=False):
        super(DiscriminatorFactorFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()

        self.fc2 = nn.Linear(hidden_size, 1)

        self.out_nonlinearity = nn.Sigmoid() if with_sigmoid else null

    def forward(self, h1, h2):

        # make sure both inputs are 2d. We assume that one is 4d and the other 2d
        if len(h1.size()) == 4:
            h1 = h1.view(h2.size())
        if len(h2.size()) == 4:
            h2 = h2.view(h1.size())
        if len(h1.size()) == 4:
            raise AssertionError("Something is amiss; both input layers were 4d")

        x = torch.cat([h1, h2], dim=1)

        out = self.fc1(x)

        out = self.relu(out)
        out = self.fc2(out)
        out = self.out_nonlinearity(out)

        return out


class DiscriminatorFactorConv(nn.Module):
    """To discriminate between two layers separated by a convolutional operation.

    The insight here is we only need to discriminate between those pairs of a lower and higher layer that,
    under the Conv2d or ConvTranpose2d, would map to one another.

    So we convolve over and look at those, expand them into a large inner_channel, and then collapse.
    (The collapsing is a convolved linear layer, i.e. Conv1d)
    The output is then the mean of all of the 'discriminators' represented by each step of the conv.

    [conv_kernel, conv_stride, conv_pad] = the parameters of the Conv2d in the inference operation between
    the two layers (or equivalently the ConvTranspose2d of the generator)

    inner_channels_per_layer: how many hidden dimension channel to allocate to each layer.

    with_sigmoid = Boolean. Whether to apply a sigmoid to the output of each inner Discriminator, as required when
                        averaging (ensembling) multiple discriminators in a standard GAN with BCELoss"""

    def __init__(self, h1_channels, h2_channels,
                 conv_kernel, conv_stride, conv_pad, inner_channels_per_layer,
                 with_sigmoid=False):
        super(DiscriminatorFactorConv, self).__init__()
        self.conv_over_h1 = nn.Conv2d(h1_channels, inner_channels_per_layer,
                                      conv_kernel, conv_stride, conv_pad)

        # the result of the upper conv must have the same 2d size as the lower conv
        # (even though the original layer sizes will be different)
        # this is an ILP problem setting (1+2*conv_upper) = (4*padding_upper+kernel_lower)
        # current impl. allows kernels of 4n-1: 3, 7, 11
        self.conv_over_h2 = nn.Conv2d(h2_channels, inner_channels_per_layer,
                                      conv_kernel, 1, int((conv_kernel + 1) / 4))

        self.conv_over_hidden_state = nn.Conv2d(2 * inner_channels_per_layer, 1, 1)

        self.relu = nn.LeakyReLU()

        self.out_nonlinearity = nn.Sigmoid() if with_sigmoid else null

    def forward(self, h1, h2):
        conved_h1 = self.conv_over_h1(h1)
        conved_h2 = self.conv_over_h2(h2)

        combined_inner_state = torch.cat([conved_h1, conved_h2], dim=1)
        combined_inner_state = self.relu(combined_inner_state)

        twoD_out = self.conv_over_hidden_state(combined_inner_state)

        bs = h1.size()[0]

        # finally take the mean over spatial dimensions. Like an ensemble over discriminators in-layer
        oneD_out = self.out_nonlinearity(twoD_out.view(bs, -1))

        return torch.mean(oneD_out, dim=1)


class Discriminator(nn.Module):
    """To discriminate between the full network state.

    Note: when calling, it takes a full dictionary of states.

    Inputs: full_architecture: a list of sizes [Input, hidden1, hidden2, z_dim]
            layer_names: a list of the names of the layers in the state_dict
            hidden_layer_size: the size of the hidden layer each discriminator
            lambda_: when using WGAN-GP, the size of the GP
            loss_type: a string. If `BCE`, then we apply a sigmoid to each sub-discriminator before averaging."""

    def __init__(self, hidden_layer_size, layer_names, lambda_=0, loss_type='wasserstein',
                 noise_dim=100,
                 image_size=28, log_intermediate_Ds=False,
                 no_backprop_through_full_cortex=False):
        super(Discriminator, self).__init__()

        self.layer_names = layer_names
        self.lambda_ = lambda_
        self.image_size = image_size

        with_sigmoid = loss_type == 'BCE'

        self.discriminator_0and1 = DiscriminatorFactorFC(image_size ** 2 + 64 * (image_size // 2) ** 2,
                                                         hidden_layer_size,
                                                         with_sigmoid=with_sigmoid)
        self.discriminator_1and2 = DiscriminatorFactorFC(128 * (image_size // 4) ** 2 + 64 * (image_size // 2) ** 2,
                                                         hidden_layer_size,
                                                         with_sigmoid=with_sigmoid)
        self.discriminator_2and3 = DiscriminatorFactorFC(128 * (image_size // 4) ** 2 + 1024,
                                                         hidden_layer_size,
                                                         with_sigmoid)
        self.discriminator_3and4 = DiscriminatorFactorFC(noise_dim + 1024,
                                                         hidden_layer_size,
                                                         with_sigmoid)

        self.Ds = [self.discriminator_0and1, self.discriminator_1and2,
                   self.discriminator_2and3, self.discriminator_3and4]

        self.log_intermediate_Ds = log_intermediate_Ds
        if log_intermediate_Ds:
            self.intermediate_Ds = {layer: [] for layer in self.layer_names}

        self.which_layers = 'all'
        self.no_backprop_through_full_cortex = no_backprop_through_full_cortex

    def forward(self, network_state_dict, inference_or_generation='inference'):
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
            h2 = network_state_dict[self.layer_names[i + 1]]

            h2 = h2.view(h2.size()[0], -1)
            h1 = h1.view(h1.size()[0], -1)

            if self.no_backprop_through_full_cortex:
                if inference_or_generation == 'inference':
                    h1 = h1.detach()
                elif inference_or_generation == 'generation':
                    h2 = h2.detach()
                else:
                    raise AssertionError("inference_or_generation should be in ['inference', 'generation']")

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

            h2g = h2g.view(h2g.size()[0], -1)
            h1g = h1g.view(h1g.size()[0], -1)

            h2i = h2i.view(h2i.size()[0], -1)
            h1i = h1i.view(h1i.size()[0], -1)

            gp = gp + calc_gradient_penalty(D, (h1i, h2i), (h1g, h2g), LAMBDA=self.lambda_)

        return gp / float(i)


def alpha_interpolate(tensor1, tensor2):
    "Returns a tensor interpolated between these two tensors, with some random about per example in the batch."
    size = tensor1.size()

    alpha = torch.rand(size[0], 1)

    # this just makes the random vector the same size as the tensor. I wish this were easier.
    if len(size) == 2:
        alpha = alpha.expand(size).to(tensor1.device)
    elif len(size) == 4:
        alpha = alpha[:, :, None, None].repeat(1, 1, size[2], size[3])
        alpha = alpha.to(tensor1.device)
    else:
        raise NotImplementedError()

    interpolated = alpha * tensor1 + ((1 - alpha) * tensor2)

    return interpolated


def calc_gradient_penalty(netD, real_data_tuple, fake_data_tuple, LAMBDA=.1):
    """A general utility function modified from a WGAN-GP implementation.

    Not pretty rn; TODO make prettier"""

    batch_size = real_data_tuple[0].size()[0]

    interpolates0 = alpha_interpolate(real_data_tuple[0], fake_data_tuple[0])
    interpolates0 = Variable(interpolates0, requires_grad=True)

    interpolates1 = alpha_interpolate(real_data_tuple[1], fake_data_tuple[1])
    interpolates1 = Variable(interpolates1, requires_grad=True)

    disc_interpolates = netD(interpolates0, interpolates1)

    gradients0, gradients1 = grad(outputs=disc_interpolates, inputs=[interpolates0, interpolates1],
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(real_data_tuple[0].device),
                                  create_graph=True, retain_graph=True, only_inputs=True)

    gradients0, gradients1 = gradients0.view(batch_size, -1), gradients1.view(batch_size, -1)

    gradient_penalty = ((gradients0.norm(2, dim=1) - 1) ** 2 +
                        (gradients1.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
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