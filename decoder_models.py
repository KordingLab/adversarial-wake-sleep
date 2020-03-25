import torch.nn as nn

class LinearDecoder(nn.Module):
    """Takes the intermediate state dict, which contains the activations of several layers,
     and trains a linear decoder from each of the layers in that dict, in parallel.

     The forward pass takes the intermediate state dict, and outputs a list of predictions.
     A CrossEntropy should be applied to each of them.
     """

    def __init__(self, image_size, noise_dim, n_classes, nc, n_filters):
        super(LinearDecoder, self).__init__()

        self.decoders = nn.ModuleList()
        self.n_features = []
        for layer in range(6):
            # get the number of features to feed to the decoder
            n_features = (nc if layer == 0 else n_filters * 2 ** (layer - 1))  # the features double each layer
            n_features *= (image_size // (2 ** layer)) ** 2  # times the edge size at this layer
            if layer == 5:
                # the top layer we handle differently
                n_features = noise_dim
            self.n_features.append(n_features)
            self.decoders.append(nn.Linear(n_features, n_classes))

    def forward(self,intermediate_state_dict):

        predictions = []
        for decoder, n_features, activations in zip(self.decoders, self.n_features, intermediate_state_dict.values()):
            out = decoder(activations.detach().view(-1,n_features))
            predictions.append(out)
        return predictions


class NonlinearDecoder(nn.Module):
    """"""
    def __init__(self, image_size, noise_dim, n_classes, nc, n_filters, hidden_dim):
        super(NonlinearDecoder, self).__init__()

        self.decoders = nn.ModuleList()
        self.n_features = []
        for layer in range(6):
            # get the number of features to feed to the decoder
            n_features = (nc if layer == 0 else n_filters * 2 ** (layer - 1))  # the features double each layer
            n_features *= (image_size // (2 ** layer)) ** 2  # times the edge size at this layer
            if layer == 5:
                # the top layer we handle differently
                n_features = noise_dim
            self.n_features.append(n_features)
            self.decoders.append(nn.Sequential(nn.Linear(n_features, hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(hidden_dim, n_classes)))

    def forward(self,intermediate_state_dict):

        predictions = []
        for decoder, n_features, activations in zip(self.decoders, self.n_features, intermediate_state_dict.values()):
            out = decoder(activations.detach().view(-1,n_features))
            predictions.append(out)
        return predictions

