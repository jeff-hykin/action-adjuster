import os
from statistics import median, mean
from copy import deepcopy

from __dependencies__.blissful_basics import FS, LazyDict
from __dependencies__.quik_config.__dependencies__ import ez_yaml
import pandas as pd

from config import config, path_to
from specific_tools.data_gathering import load_recorder
from generic_tools.plotting import graph_lines, graph_groups, xd_theme
from specific_tools import comparision_runs

experiment_number = config.experiment_number

def plot_over_time(attribute, paths=[f"{path_to.default_output_folder}/recorder.yaml"], colors=[xd_theme.blue]):
    lines = []
    for path, color in zip(paths, colors+[xd_theme.blue for _ in paths]):
        recorder_data = load_recorder(path)
        if not recorder_data:
            raise Exception(f'''couldnt read {path}''')
        records = recorder_data.records
        records_with_attribute = tuple( each for each in records if each.get(attribute, None) != None )
        if len(records_with_attribute) == 0:
            print(f'''{path} didnt have any records with {repr(attribute)}''')
        lines.append(
            dict(
                x_values=[ each.timestep   for each in records_with_attribute ],
                y_values=[ each[attribute] for each in records_with_attribute ],
                name=FS.basename(FS.dirname(path)),
                color=color or xd_theme.blue,
            )
        )
        
    graph_lines(
        *lines,
        title=f"{attribute} over time",
        x_axis_name=attribute,
        y_axis_name="Time",
    )
    
if __name__ == '__main__':
    plot_over_time(
        attribute="fit_points_time_seconds",
        paths=[
            f"{path_to.default_output_folder}/recorder.yaml",
            f"output.max_iter_40.ignore/recorder.yaml",
        ],
    )