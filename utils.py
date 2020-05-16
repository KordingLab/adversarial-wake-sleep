from matplotlib import pyplot as plt
from numpy import prod
from math import sqrt
import torch
from collections import OrderedDict
from torch.autograd import grad, Variable
from torch import nn
import math

def sv_img(img, savename, epoch = None, title=None,):
    npimg = img[[2,1,0]].permute(1,2,0).numpy()
    plt.imshow(npimg, interpolation='nearest')
    if epoch is not None:
        plt.title("Epoch {}".format(epoch))
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(savename, dpi=300)

def get_gradient_stats(net):
    """Returns the maximum of the gradients in this network. Useful to set proper weight clipping"""
    maximum = 0

    for p in net.parameters():
        maximum = max(maximum, p.grad.data.max())

    return  maximum

def KLfromSN(x):
    """Assumes the inputs are batched samples from a Gaussian dist. Calculates the KL of this dist. from a mean=0, var=1 dist.
    """
    sig = torch.var(x, 0)
    mu2 = torch.mean(x, 0) ** 2
    log_sig = torch.log(sig + 1e-12)
    mu2_m1 = mu2 - 1

    out = .5 * torch.sum(sig + mu2_m1 - log_sig)

    return out

def to_variable(intermediate_state_dict):
    v_state_dict = OrderedDict()
    for i, (layer, activations) in enumerate(intermediate_state_dict.items()):
        v_state_dict[layer] = Variable(activations.detach(), requires_grad=True)
    return v_state_dict


def get_detached_state_dict(state_dict):
    detached_dict = {k: None if (v is None) else v.detach()
                     for k, v in state_dict.items()}
    return detached_dict


def get_gradient_penalty(discriminator, state_dict, lamda = 1, p=1, ):
    """Calculate the gradient of the discriminator's output w/r/t inputs"""

    state_dict = to_variable(state_dict)

    d = discriminator(state_dict)
    bs = d.size(0)

    gradients = grad(outputs=d, inputs=[s for _,s in state_dict.items()],
                     grad_outputs=torch.ones(d.size()).to(d.device),
                     create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)

    all_gs = torch.cat([g.view(bs, -1) for g in gradients if g is not None],dim=1)
    gp =  ((all_gs.norm(p, dim=1) - 1) ** p).mean()

    return gp * lamda


def get_gradient_penalty_inputs(readout_discriminator, inference_net, input_data, lamda=10, p=2, ):
    """Calculate the gradient of the discriminator's output w/r/t inputs,
    and ensure the encoder changes accordingly too"""

    input_data = Variable(input_data.detach(), requires_grad=True)
    inference_layer = inference_net(input_data, to_layer=4, update_states=False)

    d = readout_discriminator(inference_layer)
    bs = d.size(0)

    gradients = grad(outputs=d, inputs=input_data,
                     grad_outputs=torch.ones(d.size()).to(d.device),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.view(gradients.size()[0], -1).norm(p, dim=1) - 1) ** p).mean()

    return gp * lamda

def stdDev(x):
    r"""
    Add a standard deviation channel to the current layer. Assumes a linear layer
    In other words:
        1) Compute the standard deviation of the features over the minibatch
        2) Get its mean, over all channels
        3) expand the layer and concatenate it
    Args:
        - x (tensor): previous layer
    """
    size = x.size()
    y = torch.var(x, 0)
    y = torch.sqrt(y + 1e-8)
    y = y.view(-1)
    y = torch.mean(y)
    y = torch.ones(size[0], 1).to(x.device) * y
    return torch.cat([x, y], dim=1)


def stdDev_conv(x):
    r"""
    Add a standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) concatenate it with the input
    Args:
        - x (tensor): previous layer
    """
    size = x.size()
    y = torch.var(x, 0)
    y = torch.sqrt(y + 1e-8)
    y = torch.mean(y, 0)
    y = torch.ones(size[0], 1, size[2], size[3]).to(x.device) * y

    return torch.cat([x, y], dim=1)


def gen_surprisal(inf_state_dict, generator, sigma2, criterion, surprisal=None,
                  detach_inference=True):
    ML_loss = 0
    sig = nn.Sigmoid()
    for i, G in enumerate(generator.listed_modules):

        lower_h = inf_state_dict[generator.layer_names[i]]
        upper_h = inf_state_dict[generator.layer_names[i + 1]]

        if detach_inference:
            lower_h = lower_h.detach()
            upper_h = upper_h.detach()

        x = G(upper_h)
        x = generator.activations[generator.layer_names[i + 1]](x)

        if surprisal is not None:
            scale = 2 * sig(-surprisal.detach().view(-1))
            loss = criterion(x, lower_h).view(lower_h.size(0), -1).mean(dim=1)
            loss = (loss * scale).mean()
        else:
            loss = criterion(x, lower_h)

        ML_loss = ML_loss + loss

    return ML_loss

def get_pixelwise_channel_norms(inference, generator):
    """This can be used to implement a 'soft' divisive normalization (where the 'hard' version is
    that which is implemented in Karras et al.). It can also be used for logging & debugging.

    Logs the total value by layer as a tuple (inferred norm, generated norm)

    Returns the pixel-wise distance from 0 (as a cost) to be minimized)"""

    epsilon = 1e-8
    mse = nn.MSELoss()
    total_distance_from_1 = 0

    for i, layer in enumerate(inference.layer_names):
        inferred_state = inference.intermediate_state_dict[layer]
        should_be = math.sqrt(list(inferred_state.size())[1])

        inferred_state_pixel_norm = (((inferred_state ** 2).mean(dim=1) + epsilon).sqrt())
        inferred_state_pixel_norm_dist_from_1 = mse(inferred_state_pixel_norm,
                                                    should_be * torch.ones_like(inferred_state_pixel_norm))
        total_distance_from_1 = inferred_state_pixel_norm_dist_from_1 + total_distance_from_1

    return total_distance_from_1

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