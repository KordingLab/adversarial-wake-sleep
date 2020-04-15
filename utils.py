from matplotlib import pyplot as plt
from numpy import prod
from math import sqrt
import torch
from torch.autograd import Variable, grad

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


def get_gradient_penalty(discriminator, cortex, lamda, i_or_g, only_input =False, only_output = False):
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

    gp = 0
    for i, (layer, activations) in enumerate(state_dict.items()):
        if only_output and i<5:
            continue
        elif only_input and i>0:
            continue

        x = Variable(activations.detach(), requires_grad = True)
        # pass through the remaining layers. if i==5 nothing is done here
        for F in cortex.inference.listed_modules[i:]:
            x = F(x)

        d = discriminator(x)
        bs = d.size(0)

        gradients = grad(outputs=d, inputs=x,
                         grad_outputs=torch.ones(d.size()).to(activations.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(bs, -1)

        gp = gp + ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gp * lamda
