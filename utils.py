from matplotlib import pyplot as plt

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