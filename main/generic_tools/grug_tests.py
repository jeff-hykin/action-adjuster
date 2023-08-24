from super_hash import super_hash
from warnings import warn
import ez_yaml
from  blissful_basics import FS, bytes_to_valid_string, valid_string_to_bytes, indent
from generic_tools import geometry
import pickle
import sys

# Version 0.1
    # add the replay mode
# Version 1.0
    # add CLI tools
# Version 2.0
    # add `additional_inputs` in the decorator
    # add file path args to the decorator that create file copies, then inject/replace the path arguments
    # fix the problem of tuples getting converted to lists (changes hash) when yamlizing

from ez_yaml import yaml

@yaml.register_class
class YamlPickled:
    yaml_tag = "!python/pickled"
    delimiter = "|"
    
    def __init__(self, value):
        self.value = value
    
    @classmethod
    def from_yaml(cls, constructor, node):
        string = node.value[node.value.index(YamlPickled.delimiter)+1:]
        # node.value is the python-value
        return pickle.loads(valid_string_to_bytes(string))
    
    @classmethod
    def to_yaml(cls, representer, data):
        prefix = f"{type(data.value)} Couldn't be yaml-seralized so it was pickled first".replace(YamlPickled.delimiter, "")
        # value needs to be a string (or some other yaml-primitive)
        return representer.represent_scalar(
            tag=cls.yaml_tag,
            value=prefix + YamlPickled.delimiter + bytes_to_valid_string(
                pickle.dumps(data.value, protocol=4)
            ),
            style=None,
            anchor=None
        )

class GrugTest:
    _path_to_main = None
    
    def __init__(
        self,
        name="__default__",
        fully_disable=False,
        replay_inputs=False,
        record_io=True,
        verbose=False,
        project_folder="./", # FIXME: walk up until .git
        test_folder="./tests/grug_tests/", # FIXME: walk up until .git
        # use_threading=True,
    ):
        self.name           = name
        self.fully_disable  = fully_disable
        self.replay_inputs  = replay_inputs
        self.record_io      = record_io
        self.verbose        = verbose
        self.project_folder = project_folder
        self.test_folder    = test_folder
        
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
        
    # record_summary
        # name of test set
        # path to main
        # sys.path
        # import statements for all recorded values
    
    # old feature: replaying inputs
    # def _generate_import_for(func, function_name, path_of_caller):
    #     func_module = getattr(func, "__module__", None)
    #     error_prefix = f"""\n\nHi, GrugTest here\nSo for the {function_name} function\n(which I think is defined in {repr(path_of_caller)})\n"""
    #     if not func_module:
    #         raise Exception(f'''{error_prefix}I'm trying to figure out how to import it (so I can replay the inputs)\nNormally I check {function_name}.__module__ to figure out what module its from, but that value is None\n\nTo fix this just add the 'import_statement' argument, ex:\n    @grug_test(import_statement="from somewhere import my_func as grug_function")\n    def my_func():\n        pass\n''')
    #     if func_module == "__main__":
    #         if not self._path_to_main:
    #             raise Exception(f'''{error_prefix}I'm trying to figure out how to import it (so I can replay the inputs)\nIt is defined in the __main__ but I'm unable to figure out where this is.\n\nTo fix this just add the 'import_statement' argument, ex:\n    @grug_test(import_statement="from somewhere import my_func as grug_function")\n    def my_func():\n        pass\n''')
    #         else:
    #             raise Exception(f'''{error_prefix}I'm trying to figure out how to import it (so I can replay the inputs)\nIt is defined in the __main__ but I'm unable to figure out where this is.\n\nTo fix this just add the 'import_statement' argument, ex:\n    @grug_test(import_statement="from somewhere import my_func as grug_function")\n    def my_func():\n        pass\n''')
    # 
    #     if type(path_to_main) == str:
    #         self.import_main_string += f"""import sys\n"""
    #         self.import_main_string += f"""sys.path.insert(0, {repr(FS.dirname(path_to_main))})\n"""
    #         self.import_main_string += f"""import {FS.basename(path_to_main)}"""

    # 
    # decorator
    # 
    def __call__(self, *args, how_to_import=None, func_name=None, test_name=None, **kwargs):
        """
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
        source = get_path_of_caller()
        def decorator(function_being_wrapped):
            if self.fully_disable:
                # no wrapping
                return function_being_wrapped
            
            # FIXME: add replay right here
            # FIXME: record self inside grug_info
            
            if self.record_io:
                relative_path = self.test_folder+"/"+FS.make_relative_path(coming_from=self.project_folder, to=source)
                function_name = func_name or getattr(function_being_wrapped, "__name__", "<unknown_func>")
                grug_folder_for_this_func = relative_path+"/"+function_name
                FS.ensure_is_folder(grug_folder_for_this_func)
            
            def wrapper(*args, **kwargs):
                grug_is_recording = self.record_io
                inputs_were_saved = False
                if grug_is_recording:
                    # 
                    # hash the inputs
                    # 
                    try:
                        arg = (args, kwargs)
                        input_hash = super_hash(arg)[0:12] # 12 chars is plenty for being unique 
                    except Exception as error:
                        warn(f"\n\n\nFor a grug test on this function: {repr(function_name)} I tried to hash the inputs but I wasn't able to.\nHere are the input types:\n    args: {repr(tuple(type(each) for each in args))}\n    kwargs: {repr(tuple(type(each) for each in kwargs.values()))}\nAnd here's the error: {error}", category=None, stacklevel=1, source=source)
                    
                    # 
                    # save the inputs
                    # 
                    try:
                        yaml_path = grug_folder_for_this_func+f"/{input_hash}.input.yaml"
                        if not FS.exists(yaml_path):
                            # clear the way
                            FS.write(data="", to=yaml_path)
                            # if all the args are yaml-able this will work
                            try:
                                ez_yaml.to_file(
                                    obj=dict(
                                        args=args,
                                        kwargs=kwargs,
                                        pickled_args_and_kwargs=YamlPickled(arg),
                                    ),
                                    file_path=yaml_path,
                                )
                            except Exception as error:
                                # if all the args are at least pickle-able, this will work
                                converted_args = list(args)
                                converted_kwargs = dict(converted_kwargs)
                                for index,each in enumerate(converted_args):
                                    try:
                                        yaml.to_string(each)
                                    except Exception as error:
                                        converted_args[index] = YamlPickled(each)
                                for each_key, each_value in converted_kwargs.items():
                                    try:
                                        yaml.to_string(each_value)
                                    except Exception as error:
                                        converted_kwargs[each_key] = YamlPickled(each_value)
                                
                                ez_yaml.to_file(
                                    obj=dict(
                                        args=converted_args,
                                        kwargs=converted_kwargs,
                                        pickled_args_and_kwargs=YamlPickled(arg),
                                    ),
                                    file_path=yaml_path,
                                )
                        inputs_were_saved = True
                    except Exception as error:
                        raise error
                        warn(f"\n\n\nFor a grug test on this function: {repr(function_name)} I tried to seralize the inputs but I wasn't able to.\nHere are the input types:\n    args: {repr(tuple(type(each) for each in args))}\n    kwargs: {repr(tuple(type(each) for each in kwargs.values()))}\nAnd here's the error: {error}", category=None, stacklevel=1, source=source)
                
                the_error = None
                try:
                    output = function_being_wrapped(*args, **kwargs)
                except Exception as error:
                    the_error = error
                
                # 
                # save output
                # 
                if inputs_were_saved and grug_is_recording:
                    yaml_path = grug_folder_for_this_func+f"/{input_hash}.output.yaml"
                    # clear the way
                    FS.write(data="", to=yaml_path)
                    try:
                        # write the output
                        ez_yaml.to_file(
                            obj={
                                "error_output": repr(the_error),
                                "normal_output": output 
                            },
                            file_path=yaml_path,
                        )
                    except Exception as error:
                        try:
                            # try to be informative if possible
                            if type(output) == tuple:
                                ez_yaml.to_file(
                                    obj={
                                        "error_output": repr(the_error),
                                        "normal_output": tuple(
                                            YamlPickled(each)
                                                for each in output
                                        ),
                                    },
                                    file_path=yaml_path,
                                )
                            else:
                                ez_yaml.to_file(
                                    obj={
                                        "error_output": repr(the_error),
                                        "normal_output": YamlPickled(output),
                                    },
                                    file_path=yaml_path
                                )
                        except Exception as error:
                            warn(f"\n\n\nFor a grug test on this function: {repr(function_name)} I tried to seralize the output but I wasn't able to.\nHere is the output type:\n    output: {type(output)}\nAnd here's the error: {error}", category=None, stacklevel=1, source=source)
                    
                if the_error != None and not self.replay_inputs:
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
    
        
def get_path_of_caller(*paths):
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
