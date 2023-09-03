from warnings import warn
import pickle
import os
import math
import random
import functools

from .__dependencies__.ez_yaml import yaml
from .__dependencies__ import ez_yaml
from .__dependencies__.blissful_basics import FS, bytes_to_valid_string, valid_string_to_bytes, indent, super_hash, print, randomly_pick_from, stringify, to_pure
from .__dependencies__.informative_iterator import ProgressBar

# Version 1.0
    # DONE: add counting-caps (max IO for a particular function, or in-general)
    # improve to_yaml(), allow deep recursion to make as much of the structure visible as possible
        # maybe add named tuple support
        # maybe add pandas dataframe support
        # maybe add torch tensor support
    # add CLI tools
        # capture all stdout/stderr
        # run all .test.py files
    # create add_input_for(func_id, args, kwargs, input_name)
    # use threads to offload the work
    # report which tests have recordings but were not tested during replay mode (e.g. couldn't reach/find function)
    # autodetect the git folder
# Version 2.0
    # fuzzing/coverage-tools; like analyzing boolean arguments, and generating inputs for all combinations of them
    # option to record stdout/stderr of a function
    # add `additional_inputs` in the decorator
    # add file path args to the decorator that create file copies, then inject/replace the path arguments

yaml.width = 999999999999999

# 
# 
# extend yaml support
# 
# 
if True:
    # 
    # fallback for anything to be pickled
    # 
    @yaml.register_class
    class YamlPickled:
        yaml_tag = "!python/pickled"
        delimiter = ":"
        
        def __init__(self, value):
            self.value = value
        
        @classmethod
        def from_yaml(cls, constructor, node):
            string = node.value[node.value.index(YamlPickled.delimiter)+1:]
            # node.value is the python-value
            return pickle.loads(valid_string_to_bytes(string))
        
        @classmethod
        def to_yaml(cls, representer, data):
            prefix = f"{type(data.value)}".replace(YamlPickled.delimiter, "")
            if prefix.startswith("<class '") and prefix.endswith("'>"):
                prefix = prefix[8:-2]
                
            # value needs to be a string (or some other yaml-primitive)
            return representer.represent_scalar(
                tag=cls.yaml_tag,
                value=prefix + YamlPickled.delimiter + bytes_to_valid_string(
                    pickle.dumps(data.value, protocol=4)
                ),
                style=None,
                anchor=None
            )
        
    # 
    # named_tuple support
    # 
    if True:
        def is_probably_named_tuple(obj):
            return (
                isinstance(obj, tuple) and
                hasattr(obj.__class__, '_asdict') and
                hasattr(obj.__class__, '_fields')
            )
        
        named_tuple_name_registry = {}
        named_tuple_class_registry = {}
        def named_tuple_summary(named_tuple):
            the_class = named_tuple.__class__
            fields          = getattr(the_class, "_fields", None)
            field_defaults  = getattr(the_class, "_field_defaults", getattr(getattr(the_class, '__new__', None), '__defaults__', {})) # fallback is for python 3.6
            no_defaults = len(field_defaults.keys()) == 0
            if no_defaults:
                return f'{the_class.__name__}{repr(fields)}'
            else:
                fields_without_default = tuple(each for each in fields if each not in field_defaults)
                fields_with_default = tuple(f"{each_key}={repr(each_value)}" for each_key, each_value in field_defaults.items())
                fields = fields_with_default + fields_with_default
                field_summary = ",".join(fields)
                return f'{the_class.__name__}({field_summary})'
        
        def register_named_tuple(named_tuple_class, yaml_name=None):
            # already registered
            if named_tuple_class_registry.get(named_tuple_class, None):
                return named_tuple_class
                
            name = yaml_name or named_tuple_class.__name__
            if name in named_tuple_name_registry and named_tuple_class not in named_tuple_class_registry:
                named_tuple_class_registry[the_class] = None
                warn(f"(from grug_test) I try to auto-register named tuples so that they seralize nicely, however it looks like there are two named tuples that are both called {name}. Please rename one of them, or register one under a different name using:\n    from grug_test import register_named_tuple\n    register_named_tuple(SomeNamedTupleClass, 'SomeNamedTupleClass1234')")
            
            named_tuple_name_registry[name] = True
            named_tuple_class_registry[the_class] = True
            named_tuple_class.yaml_tag = f"!python/named_tuple/{name}"
            named_tuple_class.from_yaml = lambda constructor, node: named_tuple_class(**json.loads(node.value))
            named_tuple_class.to_yaml = lambda representer, object_of_this_class: representer.represent_scalar(
                tag=named_tuple_class.yaml_tag,
                value=json.dumps(object_of_this_class._asdict()),
                style=None,
                anchor=None
            )
            
            yaml.register_class(named_tuple_class)
            return named_tuple_class
    
    # 
    # todo: from dataclasses import dataclass
    #
                
            
    # 
    # numpy support
    # 
    if True:
        has_numpy = False
        try:
            import numpy
            has_numpy = True
        except Exception as error:
            pass
        
        if has_numpy:
            try:
                import numpy
                ez_yaml.yaml.Representer.add_representer(
                    numpy.ndarray,
                    lambda dumper, data: dumper.represent_sequence(tag='python/numpy/ndarray', sequence=data.tolist()), 
                )
                ez_yaml.ruamel.yaml.RoundTripConstructor.add_constructor(
                    'python/numpy/ndarray',
                    lambda loader, node: numpy.array(loader.construct_sequence(node, deep=True)),
                )
                
                # some types are commented out because I'm unsure about them loosing precision when being re-created and I didn't feel like testing to find out
                for each in [
                    # "float",
                    'double',
                    # "cfloat",
                    # 'cdouble',
                    'float8',
                    'float16',
                    'float32',
                    'float64',
                    # 'float128',
                    # 'float256',
                    # "longdouble",
                    # "longfloat",
                    # "clongdouble",
                    # "clongfloat",
                ]:
                    def _():
                        the_type = getattr(numpy, each, None)
                        if the_type:
                            the_tag = f'python/numpy/{each}'
                            ez_yaml.yaml.Representer.add_representer(
                                the_type,
                                lambda dumper, data: dumper.represent_scalar(
                                    tag=the_tag,
                                    value=str(float(data)),
                                    style=None,
                                    anchor=None
                                ),
                            )
                            ez_yaml.ruamel.yaml.RoundTripConstructor.add_constructor(
                                the_tag,
                                lambda loader, node: the_type(node.value),
                            )
                    _() # for scoping

                for each in [
                    # "intp",
                    # "uintp",
                    # "intc",
                    # "uintc",
                    # "longlong",
                    # "ulonglong",
                    "uint",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "uint128",
                    "uint256",
                    # "int",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "int128",
                    "int256",
                ]:
                    def _():
                        the_type = getattr(numpy, each, None)
                        if the_type != None:
                            the_tag = f'python/numpy/{each}'
                            ez_yaml.yaml.Representer.add_representer(
                                the_type,
                                lambda dumper, data: dumper.represent_scalar(
                                    tag=the_tag,
                                    value=str(int(data)),
                                    style=None,
                                    anchor=None
                                ),
                            )
                            ez_yaml.ruamel.yaml.RoundTripConstructor.add_constructor(
                                the_tag,
                                lambda loader, node: the_type(node.value.split(".")[0]),
                            )
                    _() # for scoping reasons
            except Exception as error:
                warn(f"\n\n(from grug_test) It looks like you have numpy so I tried to add yaml-seralization support for it but I hit this error:{error}\n\nYou can manually add yaml-seralization for numpy if you like (from grug_test import yaml, its from the ruamel.yaml library)\nHowever you don't have to, grug_test will still work, the numpy arrays will just look like an ugly binary/hex blob")
    
    # 
    # helper
    # 
    def to_yaml(obj):
        if isinstance(obj, (tuple, list)):
            return tuple(to_yaml(each) for each in obj)
        elif isinstance(obj, dict):
            return { 
                to_yaml(each_key): to_yaml(each_value)
                    for each_key, each_value in obj.items()
            }
        else:
            if is_probably_named_tuple(obj):
                register_named_tuple(obj.__class__)
            try:
                ez_yaml.to_string(obj)
                return obj
            except Exception as error:
                return YamlPickled(obj)

class GrugTest:
    """
        Example:
            grug_test = GrugTest(
                project_folder=".",
                test_folder="./tests/grug_tests",
                fully_disable=os.environ.get("PROD")!=None,
                record_io=os.environ.get("DEV")!=None,
            )
            
            @grug_test
            def add_nums(a,b):
                return a + b + 1

            # normal usage
            for a,b in zip(range(10), range(30, 40)):
                add_nums(a,b)
    """
    overflow_strats = [ 'keep_old', 'delete_random', ]
    input_file_extension = ".input.yaml"
    output_file_extension = ".output.yaml"
    def __init__(
        self,
        name="__default__",
        fully_disable=False,
        replay_inputs=False,
        record_io=True,
        verbose=False,
        max_io_per_func=None,
        overflow_strat="keep_old",
        project_folder="./", # FIXME: walk up until .git
        test_folder="./tests/grug_tests/", # FIXME: walk up until .git
        # use_threading=True,
    ):
        self.name            = name
        self.fully_disable   = fully_disable
        self.replay_inputs   = replay_inputs
        self.record_io       = record_io
        self.verbose         = verbose
        self.overflow_strat  = overflow_strat
        self.max_io_per_func = max_io_per_func if max_io_per_func != None else math.inf
        self.project_folder  = project_folder
        self.test_folder     = test_folder
        
        # 
        # setup grug info
        # 
        grug_info_path = f"{test_folder}/{self.name}.grug_info.yaml"
        self.grug_info = {}
        grug_info_string = FS.read(grug_info_path)
        if grug_info_string:
            self.grug_info = ez_yaml.to_object(string=grug_info_string)
            if type(self.grug_info) != dict:
                self.grug_info = {}
                warn(f"The grug_info at {repr(grug_info_path)} seems to be corrupted, it will be ignored")
        
        self.grug_info = {
            **self.grug_info,
            "functions_with_tests": self.grug_info.get("functions_with_tests", []),
        }
        self.has_been_tested = { name: False for name in self.grug_info["functions_with_tests"] }
    
    # 
    # decorator
    # 
    def __call__(self, *args, save_to=None, func_name=None, max_io=None, record_io=None, additional_io_per_run=None, **kwargs):
        """
        Example:
            grug_test = GrugTest(
                project_folder=".",
                test_folder="./tests/grug_tests",
                fully_disable=os.environ.get("PROD")!=None,
                record_io=os.environ.get("DEV")!=None,
            )
            
            @grug_test
            def add_nums(a,b):
                return a + b + 1

            # normal usage
            for a,b in zip(range(10), range(30, 40)):
                add_nums(a,b)
        
        """
        if record_io == None:
            record_io = self.record_io
        if max_io == None:
            max_io = self.max_io_per_func
        if additional_io_per_run == None:
            additional_io_per_run = math.inf
        
        source = _get_path_of_caller()
        def decorator(function_being_wrapped):
            nonlocal max_io
            nonlocal save_to
            if self.fully_disable:
                # no wrapping
                return function_being_wrapped
            
            function_name = None
            grug_folder_for_this_func = None
            decorator.record_io = record_io
            decorator.replaying_inputs = False
            if self.record_io or self.replay_inputs:
                # 
                # setup name/folder
                # 
                if not save_to:
                    relative_path_to_function = FS.normalize(FS.make_relative_path(coming_from=self.project_folder, to=source))
                    relative_path_to_test = self.test_folder+"/"+relative_path_to_function
                    save_to = relative_path_to_test
                function_name = func_name or getattr(function_being_wrapped, "__name__", "<unknown_func>")
                function_id = f"{relative_path_to_function}:{function_name}"
                grug_folder_for_this_func = save_to+"/"+function_name
                if function_id not in self.grug_info["functions_with_tests"]:
                    self.grug_info["functions_with_tests"].append(function_id)
                
                FS.ensure_is_folder(grug_folder_for_this_func)
                input_files = [ each for each in FS.list_file_paths_in(grug_folder_for_this_func) if each.endswith(self.input_file_extension) ]
                # convert additional_io_per_run to a max_io value
                if additional_io_per_run > 0 and max_io > len(input_files):
                    max_io = min(max_io, len(input_files) + additional_io_per_run)
                
                # 
                # replay inputs
                # 
                if self.replay_inputs and not self.has_been_tested.get(function_id, False):
                    if self.verbose: print(f"replaying inputs for: {function_name}")
                    decorator.replaying_inputs = True
                    original_record_io_value = decorator.record_io
                    for progress, path in ProgressBar(input_files, disable_logging=not self.verbose):
                        progress.text = f" loading: {FS.basename(path)}"
                        try:
                            args, kwargs = ez_yaml.to_object(file_path=path)["pickled_args_and_kwargs"]
                            output, the_error = self.record_output(
                                function_being_wrapped,
                                args,
                                kwargs,
                                path=path[0:-len(self.input_file_extension)] + self.output_file_extension,
                                function_name=function_name,
                                source=source,
                                verbose=True,
                            )
                        except Exception as error:
                            if self.verbose:
                                print(f"corrupted_input: {path}\n    {error}")
                            else:
                                warn(f"corrupted_input: {path}\n    {error}")
                    decorator.replaying_inputs = False
                    self.has_been_tested[function_id] = True
            
            @functools.wraps(function_being_wrapped) # fixes the stack-trace to make the decorator invisible
            def wrapper(*args, **kwargs):
                # normal run
                if decorator.replaying_inputs or not decorator.record_io:
                    return function_being_wrapped(*args, **kwargs)
                
                # when overflowing, with 'keep_old' just avoid saving io (even though technically we might want to update the output of an existing one)
                is_overflowing = len(input_files) >= max_io
                shouldnt_save_new_io = is_overflowing and self.overflow_strat == 'keep_old'
                if shouldnt_save_new_io:
                    return function_being_wrapped(*args, **kwargs)
                
                # 
                # hash the inputs
                #
                input_hash = None
                try:
                    arg = (args, kwargs)
                    input_hash = super_hash(arg)[0:12] # 12 chars is plenty for being unique 
                except Exception as error:
                    error_message = f"\n\n\nFor a grug test on this function: {repr(function_name)}\n" + indent(f"I tried to hash the inputs but I wasn't able to.\nHere are the input types:\n    args: {indent(stringify(tuple(type(each) for each in args)))}\n    kwargs: {indent(stringify({ key: type(value) for key, value in kwargs.items()}))}\n\nAnd here's the error:\n{indent(error)}\n")
                    warn(error_message, category=None, stacklevel=1, source=source)
                    # run function like normal
                    return function_being_wrapped(*args, **kwargs)
                
                input_file_path  = grug_folder_for_this_func+f"/{input_hash}{self.input_file_extension}"
                output_file_path = grug_folder_for_this_func+f"/{input_hash}{self.output_file_extension}"
                
                try:
                    # 
                    # input limiter
                    # 
                    input_already_existed = FS.is_file(input_file_path)
                    if is_overflowing and not input_already_existed and self.overflow_strat == 'delete_random':
                        input_to_delete  = randomly_pick_from(input_files)
                        output_to_delete = input_to_delete[0:-len(self.input_file_extension)]+self.output_file_extension
                        FS.remove(input_to_delete)
                        FS.remove(output_to_delete)
                        input_files.remove(input_to_delete)
                    
                    # 
                    # save the inputs
                    # 
                    if not input_already_existed:
                        FS.ensure_is_folder(FS.parent_path(input_file_path))
                        # encase its a folder for some reason
                        FS.remove(input_file_path)
                        # if all the args are yaml-able this will work
                        ez_yaml.to_file(
                            file_path=input_file_path,
                            obj=dict(
                                args=to_yaml(args),
                                kwargs=to_yaml(kwargs),
                                pickled_args_and_kwargs=YamlPickled(arg),
                            ),
                        )
                        input_files.append(input_file_path)
                except Exception as error:
                    FS.remove(input_file_path)
                    warn(f"\n\n\nFor a grug test on this function: {repr(function_name)}\n"+indent(f"I tried to seralize the inputs but I wasn't able to.\nHere are the input types:\n    args: {indent(stringify(tuple(type(each) for each in args)))}\n    kwargs: {indent(stringify({ key: type(value) for key, value in kwargs.items()}))}\n\nAnd here's the error:\n{indent(error)}\n"), category=None, stacklevel=1, source=source)
                    # run function like normal
                    return function_being_wrapped(*args, **kwargs)
                
                # 
                # save the output
                # 
                output, the_error = self.record_output(
                    function_being_wrapped,
                    args,
                    kwargs,
                    path=output_file_path,
                    function_name=function_name,
                    source=source,
                    verbose=self.verbose,
                )
                
                # raise errors like normal
                if the_error != None:
                    raise the_error
                
                return output
            return wrapper
        
        # this handles
        # @grug_test
        # def thign(): pass
        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])
        # this handles
        # @grug_test(options=somethin)
        # def thign(): pass
        else:
            return decorator
    
    @staticmethod
    def record_output(func, args, kwargs, path, function_name, source, verbose):
        the_error = None
        output = None
        try:
            output = func(*args, **kwargs)
        except Exception as error:
            the_error = error
        
        # clear the way (generates parent folders if needed)
        FS.ensure_is_folder(FS.parent_path(path))
        # encase its a folder for some reason
        FS.remove(path)
        try:
            ez_yaml.to_file(
                file_path=path,
                obj={
                    "error_output": repr(the_error),
                    "normal_output": to_yaml(output),
                },
            )
        except Exception as error:
            message = f"\n\n\nFor a grug test on this function: {repr(function_name)}\n"+ indent(f"I tried to seralize the output but I wasn't able to.\nHere is the output type:\n    output: {type(output)}\nAnd here's the error: {indent(error)}\n")
            if verbose:
                print(message)
            else:
                warn(message, category=None, stacklevel=1, source=source)
    
        return output, the_error
        
def _get_path_of_caller(*paths):
    import os
    import inspect
    
    intial_cwd = os.getcwd()
    # https://stackoverflow.com/questions/28021472/get-relative-path-of-caller-in-python
    try:
        frame = inspect.stack()[2]
        module = inspect.getmodule(frame[0])
        directory = module.__file__
    # if inside a repl (error =>) assume that the working directory is the path
    except (AttributeError, IndexError) as error:
        directory = cwd
    
    if FS.is_absolute_path(directory):
        return FS.join(directory, *paths)
    else:
        # See note at the top
        return FS.join(intial_cwd, directory, *paths)