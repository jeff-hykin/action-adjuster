import os
from statistics import median, mean
from copy import deepcopy

from __dependencies__.blissful_basics import FS, LazyDict
from __dependencies__.quik_config.__dependencies__ import ez_yaml
import pandas as pd

from config import config, path_to
from specific_tools.data_gathering import get_recorder_data
from generic_tools.plotting import graph_lines, graph_groups, xd_theme

# action_adjuster.max_history_size

experiment_name = "NOISE=MEDIUM,ADVERSITY=STRONG"

groups = dict(
    no_adjuster=dict(
        folder_name_must_include="5.@NO_ADJUSTER",
        summary_filter=lambda data: "ADVERSITY=STRONG" in data["selected_profiles"] and "NOISE=MEDIUM" in data["selected_profiles"],
        color=xd_theme.red,
        lines=[],
    ),
    normal_adjuster=dict(
        folder_name_must_include="5.@NORMAL_ADJUSTER",
        summary_filter=lambda data: "ADVERSITY=STRONG" in data["selected_profiles"] and "NOISE=MEDIUM" in data["selected_profiles"],
        color=xd_theme.blue,
        lines=[],
    ),
    perfect_adjuster=dict(
        folder_name_must_include="5.@PERFECT_ADJUSTER",
        summary_filter=lambda data: "ADVERSITY=STRONG" in data["selected_profiles"] and "NOISE=MEDIUM" in data["selected_profiles"],
        color=xd_theme.green,
        lines=[],
    ),
)

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
            y_values=[ each.line_fit_score for each in entry_data ],
            name=plot_name,
            color=group_info["color"],
        )
        group_info["lines"].append(line_data)
        lines.append(line_data)
    return lines, groups

def graph_variance_median_mean(groups, prefix=""):
    graph_name = "variance"
    graph_groups(
        groups,
        title=prefix+f"_{graph_name} {experiment_name}",
        x_axis_name="Timestep",
        y_axis_name=prefix,
        save_to="./plots/"+FS.name(__file__)+"_"+prefix+"_"+graph_name+".html",
        remove_space_below_individual=False,
        group_averaging_function=None,
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
    )

reward_lines, reward_groups = extract_accumulated_reward_as_lines(groups)
graph_variance_median_mean(
    groups=reward_groups,
    prefix="reward",
)

curve_fit_lines, curve_fit_groups = extract_curve_fit_as_lines(groups)
graph_variance_median_mean(
    groups=curve_fit_groups,
    prefix="line_fit_score",
)