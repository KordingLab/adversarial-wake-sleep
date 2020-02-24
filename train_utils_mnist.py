import torch
from torch.autograd import Variable

def engage_new_layer(epoch, cortex, optimizerG, optimizerF, optimizerD, discriminator,
                     n_epochs_per_layer=1):
    """Used in sequential_training. Every n epochs, add a higher layer to be trained upon.

    This is implemented by adding new groups to the optimizer. Additionally, and redundantly, we tell the
    discriminator which of its internal modules we should call (for efficiency we don't call modules that
    won't be used for training.)

    """
    if epoch == 0:
        discriminator.which_layers = [0]
        cortex.which_layers = [0]
    if epoch == 1 * n_epochs_per_layer:
        discriminator.which_layers.append(1)
        cortex.which_layers.append(1)

        optimizerG.add_param_group({'params': cortex.generator.generative_2to1_conv.parameters()})
        optimizerF.add_param_group({'params': cortex.inference.inference_1to2_conv.parameters()})
        optimizerD.add_param_group({'params': discriminator.discriminator_1and2.parameters()})

    if epoch == 2 * n_epochs_per_layer:
        discriminator.which_layers.append(2)
        cortex.which_layers.append(2)

        optimizerG.add_param_group({'params': cortex.generator.generative_3to2.parameters()})
        optimizerF.add_param_group({'params': cortex.inference.inference_2to3.parameters()})
        optimizerD.add_param_group({'params': discriminator.discriminator_2and3.parameters()})

    if epoch == 3 * n_epochs_per_layer:
        discriminator.which_layers.append(3)
        cortex.which_layers.append(3)

        optimizerG.add_param_group({'params': cortex.generator.generative_4to3.parameters()})
        optimizerF.add_param_group({'params': cortex.inference.inference_3to4.parameters()})
        optimizerD.add_param_group({'params': discriminator.discriminator_3and4.parameters()})

    if epoch >= 0:
        noisy_layer = 1
    if epoch >= 1 * n_epochs_per_layer:
        noisy_layer = 2
    if epoch >= 2 * n_epochs_per_layer:
        noisy_layer = 3
    if epoch >= 3 * n_epochs_per_layer:
        noisy_layer = 4
    return noisy_layer

def get_batch_of_real(test_loader):
    for imgs,_ in test_loader:
        break
    return imgs.cuda()


def generate_some_images(cortex, noise_dim, batch_size):
    noisy_toplayer = Variable(torch.randn(batch_size, noise_dim).cuda())
    generated_input = cortex.generator(noisy_toplayer)

    return generated_input