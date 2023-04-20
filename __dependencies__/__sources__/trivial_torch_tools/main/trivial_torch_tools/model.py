import functools

import torch
import torch.nn as nn
from simple_namespace import namespace

from trivial_torch_tools.generics import product, bundle, large_pickle_save, large_pickle_load, apply_to_selected


@namespace
def init():
    def to_device(device=None, attribute="hardware", constructor_arg=True):
        if device is None:
            from trivial_torch_tools.core import default_device
            device = default_device
        
        def wrapper1(function_being_wrapped):
            # wrapper2 will be the new __init__()
            def wrapper2(self, *args, **kwargs):
                current_device = device
                
                if not (attribute is None):
                    setattr(self, attribute, current_device)
                    if constructor_arg and (attribute in kwargs) and not (kwargs[attribute] is None):
                        current_device = kwargs[attribute]
                
                # call the normal __init__()
                init_output = function_being_wrapped(self, *args, **kwargs)
                
                # send this to the device if possible
                if hasattr(self, "to") and callable(self.to):
                    self.to(current_device)
                
                return init_output
            return wrapper2
        return wrapper1
    
                
    def forward_sequential_method(function_being_wrapped):
        # wrapper will be the new __init__()
        def wrapper(self, *args, **kwargs):
            
            # create method (note: no self argument)
            def forward(neuron_activations):
                
                return functools.reduce(
                    (lambda x, each_layer: each_layer.forward(x)),
                    self.children(),
                    neuron_activations
                )
            # attach method
            self.forward = forward
            
            # call original __init__()
            return function_being_wrapped(self, *args, **kwargs)
        return wrapper
    
    def save_and_load_methods(basic_attributes, model_attributes=[], path_attribute="path"):
        def wrapper1(function_being_wrapped):
            # wrapper2 will be the new __init__()
            def wrapper2(self, *args, **kwargs):
                # create methods
                def save(path=None):
                    model_data = tuple(getattr(self, each_attribute).state_dict() for each_attribute in model_attributes)
                    normal_data = tuple(getattr(self, each_attribute)               for each_attribute in basic_attributes)
                    if hasattr(self, path_attribute):
                        value = getattr(self, path_attribute)
                        if isinstance(value, str):
                            path = path or value
                    return large_pickle_save((normal_data, model_data), path)
                
                def load(path=None):
                    if hasattr(self, path_attribute):
                        value = getattr(self, path_attribute)
                        if isinstance(value, str):
                            path = path or value
                    (normal_data, model_data) = large_pickle_load(path or self.path)
                    for each_attribute, each_value in zip(basic_attributes, normal_data):
                        setattr(self, each_attribute, each_value)
                    for each_attribute, each_value in zip(model_attributes, model_data):
                        getattr(self, each_attribute).load_state_dict(each_value)
                    return self
                
                # attach methods
                self.save = save
                self.__class__.load = load
                # call original __init__()
                return function_being_wrapped(self, *args, **kwargs)
            return wrapper2
        return wrapper1
    
    def freeze_tools():
        def wrapper1(function_being_wrapped):
            # wrapper2 will be the new __init__()
            def wrapper2(self, *args, **kwargs):
                # create methods
                def freeze():
                    for child in self.children():
                        for param in child.parameters():
                            param.requires_grad = False
                
                def unfreeze():
                    for child in self.children():
                        for param in child.parameters():
                            param.requires_grad = True
                
                class WithObj(object):
                    def __init__(self_, *args, **kwargs):
                        pass
                    def __enter__(self_, *_):
                        self.freeze()
                        return self
                    def __exit__(self_, _, error, traceback):
                        self.unfreeze()
                        # normal cleanup HERE
                        if error is not None:
                            # error cleanup HERE
                            raise error
                # attach methods
                self.freeze = freeze
                self.unfreeze = unfreeze
                self.frozen = WithObj()
                
                return function_being_wrapped(self, *args, **kwargs)
            return wrapper2
        return wrapper1
    
    return locals()


@namespace
def convert_each_arg():
    def to_tensor(which_args=...):
        from trivial_torch_tools.core import to_tensor as real_to_tensor
        def wrapper1(function_being_wrapped):
            # wrapper2 will be the replacement 
            def wrapper2(self, *args, **kwargs):
                # run the converter on the selected arguments
                args, kwargs = apply_to_selected(real_to_tensor, which_args, args, kwargs)
                return function_being_wrapped(self, *args, **kwargs)
            return wrapper2
        return wrapper1
    
    def to_device(device_attribute="hardware", device=None, which_args=...):
        def wrapper1(function_being_wrapped):
            # wrapper2 will be the replacement 
            def wrapper2(self, *args, **kwargs):
                def converter(value):
                    if hasattr(value, "to") and callable(getattr(value, "to")):
                        if device:
                            return value.to(device)
                        elif hasattr(self, device_attribute):
                            self_device = getattr(self, device_attribute)
                            if self_device:
                                return value.to(self_device)
                    return value
                # run the converter on the selected arguments
                args, kwargs = apply_to_selected(converter, which_args, args, kwargs)
                return function_being_wrapped(self, *args, **kwargs)
            return wrapper2
        return wrapper1
    
    def to_batched_tensor(number_of_dimensions=4, which_args=...):
        """
            will wrap single-datapoint argument to make them appear as a batch
        """
        from trivial_torch_tools.core import to_tensor as real_to_tensor
        def wrapper1(function_being_wrapped):
            # wrapper2 will be the replacement 
            def wrapper2(self, *args, **kwargs):
                def converter(input_data):
                    # converts to torch if needed
                    input_data = real_to_tensor(input_data).type(torch.float)
                    existing_dimensions = input_data.shape[-number_of_dimensions:]
                    number_missing = number_of_dimensions - len(existing_dimensions)
                    missing_dimensions = number_missing * [1]
                    dimensions = [*missing_dimensions, *existing_dimensions]
                    dimensions[0] = -1
                    # reshape to fit in the dimensions (either add dimensions, or crush dimensions)
                    input_data = input_data.reshape(dimensions)
                    return input_data
                # run the converter on the selected arguments
                args, kwargs = apply_to_selected(converter, which_args, args, kwargs)
                return function_being_wrapped(self, *args, **kwargs)
            return wrapper2
        return wrapper1
    
    def torch_tensor_from_opencv_format(number_of_dimensions=4):
        """
            
        """
        from trivial_torch_tools.image import torch_tensor_from_opencv_format as real_torch_tensor_from_opencv_format
        def wrapper1(function_being_wrapped):
            # wrapper2 will be the replacement 
            def wrapper2(self, *args, **kwargs):
                # run the converter on the selected arguments
                args, kwargs = apply_to_selected(real_torch_tensor_from_opencv_format, which_args, args, kwargs)
                return function_being_wrapped(self, *args, **kwargs)
            return wrapper2
        return wrapper1
    
    return locals()