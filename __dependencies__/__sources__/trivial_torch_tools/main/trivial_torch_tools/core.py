from collections import OrderedDict
from simple_namespace import namespace

import torch
import torch.nn as nn
from trivial_torch_tools.generics import is_like_generator, apply_to_selected

default_seed = 1
torch.manual_seed(default_seed)

# if gpu is to be used
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import numpy
except Exception as error:
    pass

def to_tensor(an_object):
    # if already a tensor, just return
    if isinstance(an_object, torch.Tensor):
        return an_object
    # if numpy, just convert
    if numpy and isinstance(an_object, numpy.ndarray):
        return torch.from_numpy(an_object).float()
    
    # if scalar, wrap it with a tensor
    if not is_like_generator(an_object):
        return torch.tensor(an_object)
    else:
        # fastest (by a lot) way to convert list of numpy elements to torch tensor
        try:
            return torch.from_numpy(numpy.stack(an_object)).float()
        except Exception as error:
            pass
        # if all tensors of the same shape
        try:
            return torch.stack(tuple(an_object), dim=0)
        except Exception as error:
            pass
        # if all scalar tensors
        try:
            return torch.tensor(tuple(an_object))
        except Exception as error:
            pass
        
        # 
        # convert each element, and make sure its not a generator
        # 
        converted_data = tuple(to_tensor(each) for each in an_object)
        # now try try again 
        
        # if all tensors of the same shape
        try:
            return torch.stack(tuple(an_object), dim=0)
        except Exception as error:
            pass
        # if all scalar tensors
        try:
            return torch.tensor(tuple(an_object))
        except Exception as error:
            pass
        # 
        # fallback: reshape to fit (give error if too badly mishapen)
        # 
        size_mismatch = False
        biggest_number_of_dimensions = 0
        non_one_dimensions = None
        # check the shapes of everything
        for tensor in converted_data:
            skipping = True
            each_non_one_dimensions = []
            for index, each_dimension in enumerate(tensor.shape):
                # keep track of number of dimensions
                if index+1 > biggest_number_of_dimensions:
                    biggest_number_of_dimensions += 1
                    
                if each_dimension != 1:
                    skipping = False
                if skipping and each_dimension == 1:
                    continue
                else:
                    each_non_one_dimensions.append(each_dimension)
            
            # if uninitilized
            if non_one_dimensions is None:
                non_one_dimensions = list(each_non_one_dimensions)
            # if dimension already exists
            else:
                # make sure its the correct shape
                if non_one_dimensions != each_non_one_dimensions:
                    size_mismatch = True
                    break
        
        if size_mismatch:
            sizes = "\n".join([ f"    {tuple(each.shape)}" for each in converted_data])
            raise Exception(f'When converting an object to a torch tensor, there was an issue with the shapes not being uniform. All shapes need to be the same, but instead the shapes were:\n {sizes}')
        
        # make all the sizes the same by filling in the dimensions with a size of one
        reshaped_list = []
        for each in converted_data:
            shape = tuple(each.shape)
            number_of_dimensions = len(shape)
            number_of_missing_dimensions = biggest_number_of_dimensions - number_of_dimensions 
            missing_dimensions_tuple = (1,)*number_of_missing_dimensions
            reshaped_list.append(torch.reshape(each, (*missing_dimensions_tuple, *shape)))
        
        return torch.stack(reshaped_list).type(torch.float)


@namespace
def args():
    
    def to_device(device=default_device, which_args=...):
        def wrapper1(function_being_wrapped):
            # wrapper2 will be the replacement 
            def wrapper2(*args, **kwargs):
                def converter(value):
                    if hasattr(value, "to") and callable(getattr(value, "to")):
                        if device:
                            return value.to(device)
                    return value
                # run the converter on the selected arguments
                args, kwargs = apply_to_selected(converter, which_args, args, kwargs)
                return function_being_wrapped(*args, **kwargs)
            return wrapper2
        return wrapper1
    
    
    return locals()