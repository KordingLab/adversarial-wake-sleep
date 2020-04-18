from matplotlib import pyplot as plt
from numpy import prod
from math import sqrt
import torch
from torch.autograd import Variable, grad
import math
from collections import OrderedDict

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

def dist_from_prior(intermediate_state_dict):
    out = 0
    for i, (layer, activations) in enumerate(intermediate_state_dict.items()):
        z = activations.view(activations.size(0),-1)
        if i==0:
            continue
        elif i==5:
            out = out + KLfromSN(z)
        else:
            out = out + moments_from_prior(z)
    return out

def KLfromSN(x):
    """Assumes the inputs are batched samples from a Gaussian dist. Calculates the KL of this dist. from a mean=0, var=1 dist.


    """
    sig = torch.var(x, 0)
    mu2 = torch.mean(x, 0) ** 2
    log_sig = torch.log(sig + 1e-12)
    mu2_m1 = mu2 - 1

    out = .5 * torch.sum(sig + mu2_m1 - log_sig)

    return out

def moments_from_prior(x):
    """
    Matches the first few moments of the batch empirical distribution to an exponential with scale = 1

    """
    loss = 0
    for i in range(1, 4):
        loss = loss + torch.mean(x ** i, 0) / math.factorial(i)

    return loss

def to_variable(intermediate_state_dict):
    v_state_dict = OrderedDict()
    for i, (layer, activations) in enumerate(intermediate_state_dict.items()):
        v_state_dict[layer] = Variable(activations.detach())
    return v_state_dict


def get_gradient_penalty(discriminator, cortex, lamda, i_or_g):
    """Calculate the gradient of the discriminator's output w/r/t all activations in the state dict,
    passing them through the encoder to get there obviously,
    and return the distance of those gradients from 1.

    Gradients are accumulated into D but also the layers of E higher than the layer in question"""

    if i_or_g == 'inference':
        state_dict = cortex.inference.intermediate_state_dict
    elif i_or_g == 'generation':
        state_dict = cortex.generator.intermediate_state_dict
    else:
        raise AssertionError("Flag should be either inference or generation")

    state_dict = to_variable(state_dict)

    d = discriminator(state_dict)
    bs = d.size(0)

    gradients = grad(outputs=d, inputs=[s for _,s in state_dict.items()],
                     grad_outputs=torch.ones(d.size()).to(d.device),
                     create_graph=True, retain_graph=True, only_inputs=True)

    gp = 0
    for g in gradients:
        gp = gp + ((g.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean()

    return gp * lamda
