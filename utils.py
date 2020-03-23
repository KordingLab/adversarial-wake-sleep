from matplotlib import pyplot as plt
from numpy import prod
from math import sqrt

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