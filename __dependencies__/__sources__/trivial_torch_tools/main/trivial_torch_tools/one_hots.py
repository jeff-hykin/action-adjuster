import torch
import torch.nn as nn
from trivial_torch_tools.core import to_tensor

class OneHotifier():
    def __init__(self, possible_values):
        # convert to tuple if needed
        if not hasattr(possible_values, "__len__"):
            possible_values = tuple(possible_values)
        self.possible_values = possible_values
    
    def to_one_hot(self, value):
        index = self.possible_values.index(value)
        # TODO: better error message if value not found 
        return torch.nn.functional.one_hot(
            torch.tensor(index),
            len(self.possible_values)
        )
    
    def from_one_hot(self, vector):
        vector = to_tensor(vector)
        index_value = vector.max(0).indices
        return self.possible_values[index_value]

    @classmethod
    def tensor_from_argmax(cls, tensor):
        tensor = to_tensor(tensor)
        the_max = max(each for each in tensor)
        onehot_tensor = torch.zeros_like(tensor)
        for each_index, each_value in enumerate(tensor):
            if each_value == the_max:
                onehot_tensor[each_index] = 1
        return onehot_tensor
    
    @classmethod
    def index_tensor_from_onehot_batch(cls, tensor_batch):
        device = None
        if isinstance(tensor_batch, torch.Tensor):
            device = tensor_batch.device
        # make sure its a tensor
        tensor_batch = to_tensor(tensor_batch)
        output = tensor_batch.max(1, keepdim=True).indices.squeeze()
        # send to same device
        return output.to(device) if device else output
        
    @classmethod
    def index_from_one_hot(cls, tensor):
        # make sure its a tensor
        tensor = to_tensor(tensor)
        return tensor.max(0, keepdim=True).indices.squeeze().item()
