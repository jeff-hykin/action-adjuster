import os
from statistics import median, mean
from copy import deepcopy

from __dependencies__.blissful_basics import FS, LazyDict
from __dependencies__.quik_config.__dependencies__ import ez_yaml
import pandas as pd

from config import config, path_to
from specific_tools.data_gathering import get_recorder_data
from generic_tools.plotting import graph_lines, graph_groups, xd_theme
from specific_tools import comparision_runs

experiment_number = config.experiment_number

experiment_name = ""
groups = dict(
    no_adjuster=dict(
        folder_name_must_include=f"{experiment_number}.@NO_ADJUSTER",
        summary_filter=lambda data: True, # "ADVERSITY=STRONG" in data["selected_profiles"] and "NOISE=MEDIUM" in data["selected_profiles"],
        color=xd_theme.red,
        lines=[],
    ),
    normal_adjuster=dict(
        folder_name_must_include=f"{experiment_number}.@NORMAL_ADJUSTER",
        summary_filter=lambda data: True, # "ADVERSITY=STRONG" in data["selected_profiles"] and "NOISE=MEDIUM" in data["selected_profiles"],
        color=xd_theme.blue,
        lines=[],
    ),
    perfect_adjuster=dict(
        folder_name_must_include=f"{experiment_number}.@PERFECT_ADJUSTER",
        summary_filter=lambda data: True, # "ADVERSITY=STRONG" in data["selected_profiles"] and "NOISE=MEDIUM" in data["selected_profiles"],
        color=xd_theme.green,
        lines=[],
    ),
)

def log_scale(value):
    import math
    if value == 0:
        return 0
    if value > 0:
        return math.log(value+1)
    else:
        return -math.log((-value)+1)

def no_scale(value):
    return value

def load_group_data(groups):
    for group_name, group_info in groups.items():
        for file_name, data in get_recorder_data(group_info["folder_name_must_include"]):
            data["parent_data_snapshot"].setdefault("selected_profiles", []) # some datasets were made before this was a saved attribute
            if group_info["summary_filter"](data["parent_data_snapshot"]):
                yield (group_name, group_info, file_name, data)

def extract_accumulated_reward_as_lines(groups):
    groups = deepcopy(groups)
    lines = []
    for group_name, group_info, file_name, data in load_group_data(groups):
        # data.records[0] = {"accumulated_reward": 0, "reward": 0, "timestep": 700, "line_fit_score": -0.18230862363710315}
        plot_name = file_name.replace("@","").replace("|"," ").lower()
        reward_data = [ each for each in data.records if each.get("accumulated_reward", None) != None ]
        line_data = dict(
            x_values=[ each.timestep           for each in reward_data ],
            y_values=[ each.accumulated_reward for each in reward_data ],
            name=plot_name,
            color=group_info["color"],
        )
        group_info["lines"].append(line_data)
        lines.append(line_data)
    return lines, groups

def extract_curve_fit_as_lines(groups):
    groups = deepcopy(groups)
    lines = []
    for group_name, group_info, file_name, data in load_group_data(groups):
        # data.records[0] = {"accumulated_reward": 0, "reward": 0, "timestep": 700, "line_fit_score": -0.18230862363710315}
        plot_name = file_name.replace("@","").replace("|"," ").lower()
        entry_data = [ each for each in data.records if each.get("line_fit_score", None) != None ]
        line_data = dict(
            x_values=[ each.timestep       for each in entry_data ],
            y_values=[ log_scale(each.line_fit_score) for each in entry_data ],
            name=plot_name,
            color=group_info["color"],
        )
        group_info["lines"].append(line_data)
        lines.append(line_data)
    return lines, groups

def extract_distance_to_optimal_as_lines(groups):
    feature = "distance_to_optimal"
    scale = log_scale
    groups = deepcopy(groups)
    lines = []
    for group_name, group_info, file_name, data in load_group_data(groups):
        # data.records[0] = {"accumulated_reward": 0, "reward": 0, "timestep": 700, "line_fit_score": -0.1823086236371031, "distance_to_optimal": 0.0}
        plot_name = file_name.replace("@","").replace("|"," ").lower()
        entry_data = [ each for each in data.records if each.get(feature, None) != None ]
        line_data = dict(
            x_values=[ each.timestep       for each in entry_data ],
            y_values=[ scale(each[feature]) for each in entry_data ],
            name=plot_name,
            color=group_info["color"],
        )
        group_info["lines"].append(line_data)
        lines.append(line_data)
    return lines, groups

def graph_variance_median_mean(groups, prefix="", display=True):
    graph_name = "variance"
    graph_groups(
        groups,
        title=prefix+f"_{graph_name} {experiment_name}",
        x_axis_name="Timestep",
        y_axis_name=prefix,
        save_to="./plots/"+FS.name(__file__)+"_"+prefix+"_"+graph_name+".html",
        remove_space_below_individual=False,
        group_averaging_function=None,
        display=display,
    )
    graph_name = "median"
    graph_groups(
        groups,
        title=prefix+f"_{graph_name} {experiment_name}",
        x_axis_name="Timestep",
        y_axis_name=prefix,
        save_to="./plots/"+FS.name(__file__)+"_"+prefix+"_"+graph_name+".html",
        remove_space_below_individual=False,
        group_averaging_function=median,
        display=display,
    )
    graph_name = "mean"
    graph_groups(
        groups,
        title=prefix+f"_{graph_name} {experiment_name}",
        x_axis_name="Timestep",
        y_axis_name=prefix,
        save_to="./plots/"+FS.name(__file__)+"_"+prefix+"_"+graph_name+".html",
        remove_space_below_individual=False,
        group_averaging_function=mean,
        # y_axis_scale="log",
        display=display,
    )

def main(display=True):
    reward_lines, reward_groups = extract_accumulated_reward_as_lines(groups)
    graph_variance_median_mean(
        groups=reward_groups,
        prefix="reward",
        display=display,
    )
    
    curve_fit_lines, curve_fit_groups = extract_curve_fit_as_lines(groups)
    graph_variance_median_mean(
        groups=curve_fit_groups,
        prefix="line_fit_score",
        display=display,
    )
    
    lines, new_groups = extract_distance_to_optimal_as_lines(groups)
    graph_variance_median_mean(
        groups=new_groups,
        prefix="distance_to_optimal",
        display=display,
    )

if __name__ == '__main__':
    main()