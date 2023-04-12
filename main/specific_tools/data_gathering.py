import ez_yaml
from blissful_basics import FS, LazyDict
import pandas as pd

from config import config, path_to

def get_recorder_data(*names):
    which_experiments = []
    if len(names) == 0:
        which_experiments = FS.list_paths_in(path_to.records)
    else:
        # case-insensive filter by basename
        which_experiments = [
            each
                for each in FS.list_paths_in(path_to.records)
                    if any([
                            each_name.lower() in FS.basename(each).lower() 
                                for each_name in names 
                        ]) 
        ]
        
    output = []
    for path in which_experiments:
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