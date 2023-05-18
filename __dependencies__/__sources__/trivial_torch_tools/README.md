# What is this?

Functions and decorators I found myself rewriting for every pytorch project

# How do I use this?

`pip install trivial-torch-tools`

```python
from trivial_torch_tools import Sequential, init
import torch.nn as nn

class Model(nn.Module):
    @init.to_device()
    # ^ does self.to() and defaults to GPU if available (uses default_device variable)
    @init.save_and_load_methods(model_attributes=["layers"], basic_attributes=["input_shape"])
    # ^ creates self.save(path=self.path) and self.load(path=self.path)
    def __init__(self):
        self.input_shape = (81,81,3)
        layers = Sequential(input_shape=self.input_shape)
        # ^ dynamically compute the output shape/size of layers (the nn.Linear below)
        layers.add_module('conv1'   , nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0))
        layers.add_module('relu1'   , nn.ReLU())
        layers.add_module('flatten' , nn.Flatten(start_dim=1, end_dim=-1))
        layers.add_module('linear1' , nn.Linear(in_features=layers.output_size, out_features=10)) 
        layers.add_module('sigmoid1', nn.Sigmoid())
        self.layers = layers
        
        # layers.output_size
        # layers.output_shape
        # layers.layer_shapes
   
# available tools
from trivial_torch_tools import *

core.default_device # defaults to cuda if available
core.to_tensor(nested_lists_of_arrays_tuples_and_more) # aggresively converts objects to tensors

# decorators for def __init__()
@model.init.to_device(device=default_device)
@model.init.save_and_load_methods(basic_attributes=[], model_attributes=[], path_attribute="path")
@model.init.forward_sequential_method
# decorators for def forward(): # or whatever 
@model.convert_each_arg.to_tensor() # Use to_tensor(which_args=[0]) to only convert first arg
@model.convert_each_arg.to_device() # Use to_device(which_args=[0]) to only convert first arg
@model.convert_each_arg.to_batched_tensor(number_of_dimensions=4) # 4 works for color images
@model.convert_each_arg.torch_tensor_from_opencv_format()

image.tensor_from_path(path)
image.pil_image_from_tensor(tensor)
image.torch_tensor_from_opencv_format(tensor_or_array)
image.opencv_tensor_from_torch_format(tensor)
image.opencv_array_from_pil_image(image_obj)

OneHotifier.tensor_from_argmax(tensor)             # [0.1,99,0,0,] => [0,1,0,0,]
OneHotifier.index_from_one_hot(tensor)             # [0,1,0,0,] => 2
OneHotifier.index_tensor_from_onehot_batch(tensor) # [[0,1,0,0,]] => [2]

import torch
converter = OneHotifier(possible_values=[ "thing0", ('thing', 1), {"thing":2} ])
converter.to_one_hot({"thing":2}) # >>> tensor([0,0,1])
converter.from_one_hot(torch.tensor([0,0,1])) # >>> {"thing":2}
```
