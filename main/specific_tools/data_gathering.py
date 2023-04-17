import ez_yaml
from blissful_basics import FS, LazyDict
import pandas as pd

from config import config, path_to

recursive_lazy_dict = lambda arg: arg if not isinstance(arg, dict) else LazyDict({ key: recursive_lazy_dict(value) for key, value in arg.items() })

path_cache = {}
def load_recorder(path):
    path = FS.make_absolute_path(path)
    if not path_cache.get(path, None):
        output = None
        if FS.is_file(path):
            data = recursive_lazy_dict(
                ez_yaml.to_object(
                    string=FS.read(path)
                )
            )
            try:
                data.config = data.parent_data_snapshot.config
            except Exception as error:
                pass
            data.records = [ LazyDict(each) for each in data.records ]
            output = data
        path_cache[path] = data
    return path_cache[path]

def get_recorder_data(*names, quiet=False):
    which_experiments = []
    file_paths = FS.list_paths_in(path_to.records)
    file_paths.sort()
    if len(names) == 0:
        which_experiments = file_paths
    else:
        # case-insensive filter by basename
        which_experiments = []
        for each_name_to_include in names:
            which_experiments += [
                each_path
                    for each_path in file_paths
                        if each_name_to_include.lower() in FS.basename(each_path).lower() and each_path not in which_experiments
            ]
    
    output = []
    for path in which_experiments:
        if not quiet:
            print(f"    parsing: {path}")
        data = load_recorder(path+"/recorder.yaml")
    
    return output