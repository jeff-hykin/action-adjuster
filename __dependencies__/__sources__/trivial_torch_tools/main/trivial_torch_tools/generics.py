def is_iterable(thing):
    # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    try:
        iter(thing)
    except TypeError:
        return False
    else:
        return True

def is_like_generator(thing):
    return is_iterable(thing) and not isinstance(thing, (str, bytes))

def flatten(value):
    flattener = lambda *m: (i for n in m for i in (flattener(*n) if is_like_generator(n) else (n,)))
    return list(flattener(value))

def flatten_once(items):
    for each in items:
        if is_like_generator(each):
            yield from each
        else:
            yield each

def product(iterable):
    from functools import reduce
    import operator
    return reduce(operator.mul, iterable, 1)

import collections.abc
def merge(old_value, new_value):
    # if not dict, see if it is iterable
    if not isinstance(new_value, collections.abc.Mapping):
        if is_iterable(new_value):
            new_value = { index: value for index, value in enumerate(new_value) }
    
    # if still not a dict, then just return the current value
    if not isinstance(new_value, collections.abc.Mapping):
        return new_value
    # otherwise get recursive
    else:
        # if not dict, see if it is iterable
        if not isinstance(old_value, collections.abc.Mapping):
            if is_iterable(old_value):
                old_value = { index: value for index, value in enumerate(old_value) }
        # if still not a dict
        if not isinstance(old_value, collections.abc.Mapping):
            # force it to be one
            old_value = {}
        
        # override each key recursively
        for key, updated_value in new_value.items():
            old_value[key] = merge(old_value.get(key, {}), updated_value)
        
        return old_value

def bundle(iterable, bundle_size):
    next_bundle = []
    for each in iterable:
        next_bundle.append(each)
        if len(next_bundle) >= bundle_size:
            yield tuple(next_bundle)
            next_bundle = []
    # return any half-made bundles
    if len(next_bundle) > 0:
        yield tuple(next_bundle)


def recursively_map(an_object, function, is_key=False):
    # base case 1 (iterable but treated like a primitive)
    if isinstance(an_object, str):
        return_value = an_object
    # base case 2 (exists because of scalar numpy/pytorch/tensorflow objects)
    if hasattr(an_object, "tolist"):
        return_value = an_object.tolist()
    else:
        # base case 3
        if not is_iterable(an_object):
            return_value = an_object
        else:
            if isinstance(an_object, dict):
                return_value = { recursively_map(each_key, function, is_key=True) : recursively_map(each_value, function) for each_key, each_value in an_object.items() }
            else:
                return_value = [ recursively_map(each, function) for each in an_object ]
    
    # convert lists to tuples so they are hashable
    if is_iterable(return_value) and not isinstance(return_value, dict) and not isinstance(return_value, str):
        return_value = tuple(return_value)
    
    return function(return_value, is_key=is_key)

def to_pure(an_object, recursion_help=None):
    # 
    # infinte recursion prevention
    # 
    top_level = False
    if recursion_help is None:
        top_level = True
        recursion_help = {}
    class PlaceHolder:
        def __init__(self, id):
            self.id = id
        def eval(self):
            return recursion_help[key]
    object_id = id(an_object)
    # if we've see this object before
    if object_id in recursion_help:
        # if this value is a placeholder, then it means we found a child that is equal to a parent (or equal to other ancestor/grandparent)
        if isinstance(recursion_help[object_id], PlaceHolder):
            return recursion_help[object_id]
        else:
            # if its not a placeholder, then we already have cached the output
            return recursion_help[object_id]
    # if we havent seen the object before, give it a placeholder while it is being computed
    else:
        recursion_help[object_id] = PlaceHolder(object_id)
    
    parents_of_placeholders = set()
    
    # 
    # main compute
    # 
    return_value = None
    # base case 1 (iterable but treated like a primitive)
    if isinstance(an_object, str):
        return_value = an_object
    # base case 2 (exists because of scalar numpy/pytorch/tensorflow objects)
    elif hasattr(an_object, "tolist"):
        return_value = an_object.tolist()
    else:
        # base case 3
        if not is_iterable(an_object):
            return_value = an_object
        else:
            if isinstance(an_object, dict):
                return_value = {
                    to_pure(each_key, recursion_help) : to_pure(each_value, recursion_help)
                        for each_key, each_value in an_object.items()
                }
            else:
                return_value = [ to_pure(each, recursion_help) for each in an_object ]
    
    # convert iterables to tuples so they are hashable
    if is_iterable(return_value) and not isinstance(return_value, dict) and not isinstance(return_value, str):
        return_value = tuple(return_value)
    
    # update the cache/log with the real value
    recursion_help[object_id] = return_value
    #
    # handle placeholders
    #
    if is_iterable(return_value):
        # check if this value has any placeholder children
        children = return_value if not isinstance(return_value, dict) else [ *return_value.keys(), *return_value.values() ]
        for each in children:
            if isinstance(each, PlaceHolder):
                parents_of_placeholders.add(return_value)
                break
        # convert all the placeholders into their final values
        if top_level == True:
            for each_parent in parents_of_placeholders:
                iterator = enumerate(each_parent) if not isinstance(each_parent, dict) else each_parent.items()
                for each_key, each_value in iterator:
                    if isinstance(each_parent[each_key], PlaceHolder):
                        each_parent[each_key] = each_parent[each_key].eval()
                    # if the key is a placeholder
                    if isinstance(each_key, PlaceHolder):
                        value = each_parent[each_key]
                        del each_parent[each_key]
                        each_parent[each_key.eval()] = value
    
    # finally return the value
    return return_value

def large_pickle_load(file_path):
    """
    This is for loading really big python objects from pickle files
    ~4Gb max value
    """
    import pickle
    import os
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    output = pickle.loads(bytes_in)
    return output

def large_pickle_save(variable, file_path):
    """
    This is for saving really big python objects into a file
    so that they can be loaded in later
    ~4Gb max value
    """
    import file_system_py as FS
    import pickle
    bytes_out = pickle.dumps(variable, protocol=4)
    max_bytes = 2**31 - 1
    FS.clear_a_path_for(file_path, overwrite=True)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

from random import sample
permute = lambda a_list: sample(a_list, k=len(tuple(a_list)))

def max_index(iterable):
    max_value = max(iterable)
    return to_pure(iterable).index(max_value)

def relative_path(*filepath_peices):
    import os
    # one-liner version:
    # relative_path = lambda *filepath_peices : os.path.join(os.path.dirname(__file__), *filepath_peices)
    return os.path.join(os.path.dirname(__file__), *filepath_peices)

def apply_to_selected(func, which_args, args, kwargs):
    if which_args == ...:
        new_args = tuple(func(each) for each in args)
        new_kwargs = { each_key : func(each_value) for each_key, each_value in kwargs.items() }
        return new_args, new_kwargs
    else:
        # todo: probably make this more flexible
        which_args = tuple(which_args)
        
        new_args = []
        for index, each in enumerate(args):
            if index in which_args:
                new_args[index].append(func(each))
            else:
                new_args[index].append(each)
            
        new_kwargs = {}
        for key, value in kwargs.items():
            if key in which_args:
                new_kwargs[key].append(func(value))
            else:
                new_kwargs[key].append(value)
        
        return new_args, new_kwargs
