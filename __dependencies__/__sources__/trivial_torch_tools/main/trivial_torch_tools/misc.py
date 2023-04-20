from collections import OrderedDict
import torch
import torch.nn as nn

def batch_input_and_output(inputs, outputs, batch_size):
    from trivial_torch_tools.generics import bundle
    batches = zip(bundle(inputs, batch_size), bundle(outputs, batch_size))
    for each_input_batch, each_output_batch in batches:
        yield to_tensor(each_input_batch), to_tensor(each_output_batch)

def unnormalize(mean, std, image):
    import torchvision.transforms as transforms
    normalizer = transforms.Normalize((-mean / std), (1.0 / std))
    return normalizer(image)

# returns list of tensor sizes
def layer_output_shapes(network, input_shape, device=None):
    from trivial_torch_tools.core import default_device
    # convert OrderedDict's to just lists
    if isinstance(network, OrderedDict):
        network = list(network.values())
    # convert lists to sequences
    if isinstance(network, list):
        network = nn.Sequential(*network)
    
    # run a forward pass to figure it out
    neuron_activations = torch.ones((1, *input_shape))
    neuron_activations = neuron_activations.to(device) if not (device is None) else neuron_activations
    sizes = []
    with torch.no_grad():
        try:
            for layer in network:
                # if its not a loss function
                if not isinstance(layer, torch.nn.modules.loss._Loss):
                    neuron_activations = layer.forward(neuron_activations)
                    sizes.append(neuron_activations.size())
        except Exception as error:
            neuron_activations = neuron_activations.to(default_device)
            for layer in network:
                # if its not a loss function
                if not isinstance(layer, torch.nn.modules.loss._Loss):
                    layer = layer.to(default_device)
                    neuron_activations = layer.forward(neuron_activations)
                    sizes.append(neuron_activations.size())
        
    return sizes
