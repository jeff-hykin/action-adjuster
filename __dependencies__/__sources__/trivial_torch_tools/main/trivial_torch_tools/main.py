import functools

import torch.nn as nn
from .generics import product, bundle, large_pickle_save, large_pickle_load
from .core import default_device, to_tensor, args
from .misc import layer_output_shapes
from .model import init, convert_each_arg
from .one_hots import OneHotifier
from . import image

class Sequential(nn.Sequential):
    @init.forward_sequential_method
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__(*args)
        self.input_shape = kwargs.get("input_shape", None)
    
    @property
    def layer_shapes(self):
        return layer_output_shapes(self, self.input_shape)
    
    @property
    def output_shape(self):
        return self.input_shape if len(self) == 0 else self.layer_shapes[-1]
    
    @property
    def output_size(self):
        total = 1
        for each in self.output_shape:
            total *= each
        return total