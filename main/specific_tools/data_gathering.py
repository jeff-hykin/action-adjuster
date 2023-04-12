import ez_yaml
from blissful_basics import FS, LazyDict
import pandas as pd

from config import config, path_to

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
        recorder_path = path+"/recorder.yaml"
        if FS.is_file(recorder_path):
            data = LazyDict(
                ez_yaml.to_object(
                    string=FS.read(recorder_path)
                )
            )
            data.records = [ LazyDict(each) for each in data.records ]
            output.append(
                (
                    FS.basename(path),
                    data,
                )
            )
    
    return output