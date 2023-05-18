from collections import OrderedDict
import torch
import torch.nn as nn
from .__dependencies__.super_hash import super_hash

def batch_input_and_output(inputs, outputs, batch_size):
    from .generics import bundle
    batches = zip(bundle(inputs, batch_size), bundle(outputs, batch_size))
    for each_input_batch, each_output_batch in batches:
        yield to_tensor(each_input_batch), to_tensor(each_output_batch)

def unnormalize(mean, std, image):
    import torchvision.transforms as transforms
    normalizer = transforms.Normalize((-mean / std), (1.0 / std))
    return normalizer(image)

# returns list of tensor sizes
def layer_output_shapes(network, input_shape, device=None):
    from .core import default_device
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

def _string_hash_to_number(string):
    original = string.encode()
    used_indicies = set()
    number = 0
    base = 256
    for index, number in enumerate(string.encode()):
        number += (base**index) * number
    return number
    
class DeterministicTorchRng:
    max_pytorch_seed_size = 2**64-1
    def __init__(self, *args, **kwargs):
        self.temp_rng_seed = _string_hash_to_number(super_hash(args)) % DeterministicTorchRng.max_pytorch_seed_size
    
    def __enter__(self):
        self.original_rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.temp_rng_seed)
        return None
    
    def __exit__(self, _, error, traceback):
        # normal cleanup HERE
        torch.random.set_rng_state(self.original_rng_state)
        if error is not None:
            # error cleanup HERE
            raise error