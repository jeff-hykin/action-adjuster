from super_hash import super_hash
from warnings import warn
import ez_yaml


def path_of_caller(*paths):
    import os
    import inspect
    
    cwd = os.getcwd()
    # https://stackoverflow.com/questions/28021472/get-relative-path-of-caller-in-python
    try:
        frame = inspect.stack()[2]
        module = inspect.getmodule(frame[0])
        directory = os.path.dirname(module.__file__)
    # if inside a repl (error =>) assume that the working directory is the path
    except (AttributeError, IndexError) as error:
        directory = cwd
    
    if is_absolute_path(directory):
        return join(directory, *paths)
    else:
        # See note at the top
        return join(intial_cwd, directory, *paths)

class GrugTest:
    production_override = False
    replay_inputs = False
    record_outputs = True
    record_inputs = False
    verbose = False
    project_folder = "./tests/grug_tests/" # FIXME: walk up until .git
    test_folder = "./tests/grug_tests/" # FIXME: walk up until .git

test_counts = {}
def grug_test():
    def decorator(function_being_wrapped):
        if GrugTest.production_override:
            # no wrapping
            return function_being_wrapped
        if GrugTest.record_inputs:
            source = path_of_caller()
            relative_path = GrugTest.test_folder+"/"+FS.make_relative_path(coming_from=source, to=GrugTest.project_folder)
            function_name = getattr(function_being_wrapped, "__name__", "<unknown_func>")
            grug_folder_for_this_func = relative_path+"/"+function_name
            FS.ensure_is_folder(grug_folder_for_this_func)
        
        class YamlFailed: pass
        
        def wrapper(*args, **kwargs):
            grug_is_testing = GrugTest.record_inputs or GrugTest.record_outputs
            if grug_is_testing:
                try:
                    arg = (args, kwargs)
                    input_hash = super_hash(arg)
                except Exception as error:
                    warn(f"\n\n\nFor a grug test on this function: {repr(function_name)} I tried to hash the inputs but I wasn't able to.\nHere are the input types:\n    args: {repr(tuple(type(each) for each in args))}\n    kwargs: {repr(tuple(type(each) for each in kwargs.values()))}\nAnd here's the error: {error}", category=None, stacklevel=1, source=source)
                    
            if GrugTest.record_inputs:
                try:
                    yaml_path = grug_folder_for_this_func+f"/{input_hash}.input.yaml"
                    pickle_path = grug_folder_for_this_func+f"/{input_hash}.input.pickle"
                    if not (FS.exists(yaml_path) or FS.exits(pickle_path)):
                        yamlized_function_input = YamlFailed
                        try:
                            yamlized_function_input = ez_yaml.to_obj(arg)
                            FS.write(data=yamlized_function_input, to=yaml_path)
                        except Exception as error:
                            large_pickle_save((args, kwargs), pickle_path)
                except Exception as error:
                    warn(f"\n\n\nFor a grug test on this function: {repr(function_name)} I tried to seralize the inputs but I wasn't able to.\nHere are the input types:\n    args: {repr(tuple(type(each) for each in args))}\n    kwargs: {repr(tuple(type(each) for each in kwargs.values()))}\nAnd here's the error: {error}", category=None, stacklevel=1, source=source)
            
            the_error = None
            try:
                output = function_being_wrapped(*args, **kwargs)
            except Exception as error:
                the_error = error
                
            if GrugTest.record_outputs:
                yaml_path = grug_folder_for_this_func+f"/{input_hash}.output.yaml"
                pickle_path = grug_folder_for_this_func+f"/{input_hash}.output.pickle"
                grug_output = { "error_output": repr(the_error), "normal_output": output }
                try:
                    yamlized_function_output = ez_yaml.to_obj(grug_output)
                    FS.write(data=yamlized_function_output, to=yaml_path)
                except Exception as error:
                    try:
                        large_pickle_save(grug_output, pickle_path)
                    except Exception as error:
                        warn(f"\n\n\nFor a grug test on this function: {repr(function_name)} I tried to seralize the output but I wasn't able to.\nHere is the output type:\n    output: {type(output)}\nAnd here's the error: {error}", category=None, stacklevel=1, source=source)
                
            if the_error != None and not GrugTest.replay_inputs:
                raise the_error
            
            return output
        return wrapper
    return decorator