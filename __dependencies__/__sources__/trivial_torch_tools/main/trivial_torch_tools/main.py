import functools

import torch.nn as nn
from trivial_torch_tools.generics import product, bundle, large_pickle_save, large_pickle_load
from trivial_torch_tools.core import default_device, to_tensor, args
from trivial_torch_tools.misc import layer_output_shapes
from trivial_torch_tools.model import init, convert_each_arg
from trivial_torch_tools.one_hots import OneHotifier
import trivial_torch_tools.image as image

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