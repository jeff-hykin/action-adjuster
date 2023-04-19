import os
from statistics import median, mean
from copy import deepcopy

import ez_yaml
from blissful_basics import FS, LazyDict
import pandas as pd

from config import config, path_to
from specific_tools.data_gathering import get_recorder_data
from generic_tools.plotting import graph_lines, xd_theme

# action_adjuster.max_history_size

experiment_name = "NOISE=NONE,ADVERSITY=STRONG"

groups = LazyDict(
    no_adjuster=LazyDict(
        folder_name_must_include="@NO_ADJUSTER",
        summary_filter=lambda data: "ADVERSITY=STRONG" in data.selected_profiles and "NOISE=NONE" in data.selected_profiles,
        color=xd_theme.red,
        lines=[],
    ),
    normal_adjuster=LazyDict(
        folder_name_must_include="@NORMAL_ADJUSTER",
        summary_filter=lambda data: "ADVERSITY=STRONG" in data.selected_profiles and "NOISE=NONE" in data.selected_profiles,
        color=xd_theme.blue,
        lines=[],
    ),
    perfect_adjuster=LazyDict(
        folder_name_must_include="@PERFECT_ADJUSTER",
        summary_filter=lambda data: "ADVERSITY=STRONG" in data.selected_profiles and "NOISE=NONE" in data.selected_profiles,
        color=xd_theme.green,
        lines=[],
    ),
)

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
            color=group_info.color,
        )
        group_info.lines.append(line_data)
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
            color=group_info.color,
        )
        group_info.lines.append(line_data)
        lines.append(line_data)
    return lines, groups

def load_group_data(groups):
    for group_name, group_info in groups.items():
        for file_name, data in get_recorder_data(group_info.folder_name_must_include):
            data.parent_data_snapshot.setdefault("selected_profiles", []) # some datasets were made before this was a saved attribute
            if group_info.summary_filter(data.parent_data_snapshot):
                yield (group_name, group_info, file_name, data)

def create_graph(
    title,
    graph_name,
    y_axis_name,
    lines,
    groups,
    should_flatten_graph,
    should_average,
    averaging_function,
):
    # 
    # flatten
    # 
    if should_flatten_graph:
        # find the min y value for each x
        from collections import defaultdict
        per_x_value = defaultdict(lambda:[])
        for each_line in lines:
            for each_x, each_y in zip(each_line["x_values"], each_line["y_values"]):
                per_x_value[each_x].append(each_y)
        min_per_x = {}
        for each_x, values in per_x_value.items():
            min_per_x[each_x] = min(values)
        # flatten all the data
        for each_line in lines:
            for index, (each_x, each_y) in enumerate(zip(each_line["x_values"], each_line["y_values"])):
                each_line["y_values"][index] = each_y - min_per_x[each_x]
    
    # 
    # group average
    # 
    if should_average:
        def points_to_function(x_values, y_values, method="linear"):
            values = list(zip(x_values, y_values))
            values.sort(reverse=False, key=lambda each: each[0])
            def shift_towards(*, new_value, old_value, proportion):
                if proportion == 1:
                    return new_value
                if proportion == 0:
                    return old_value
                
                difference = new_value - old_value
                amount = difference * proportion
                return old_value+amount
            
            def new_function(x_input):
                prev_x, prev_y = values[0]
                if x_input <= prev_x: # x_input is outside of the bounds
                    return prev_y 
                max_x, max_y = values[-1]
                if x_input >= max_x: # x_input is outside of the bounds
                    return max_y
                
                for each_x, each_y in values:
                    # they must not be equal, so skip
                    if each_x == prev_x:
                        continue
                    
                    if each_x == x_input:
                        return each_y
                    elif each_x > x_input > prev_x:
                        the_range = each_x - prev_x
                        relative_amount = x_input - prev_x
                        proportion = relative_amount/the_range
                        return shift_towards(new_value=each_x, old_value=prev_x, proportion=proportion)
                    
                    prev_x = each_x
                    prev_y = each_y
                
                # if its a vertical line or only has one point, this line will run
                return prev_y
                        
            return new_function
                
        
        new_lines = []
        for group_name, each_group in groups.items():
            x_values = each_group.lines[0]["x_values"]
                
            functions = [ points_to_function(each["x_values"], each["y_values"]) for each in each_group.lines ]
            y_values = [
                averaging_function([ each_function(each_x) for each_function in functions ])
                    for each_x in x_values
            ]
            new_lines.append(
                dict(
                    x_values=x_values,
                    y_values=y_values,
                    name=group_name,
                    color=each_group.color,
                )
            )
        
        lines = new_lines
    
    graph_lines(
        *lines,
        title=title,
        x_axis_name="Timestep",
        y_axis_name=y_axis_name,
        save_to="./plots/"+FS.name(__file__)+"_"+graph_name+".html",
    )

def graph_variance_median_mean(lines, groups, prefix=""):
    create_graph(
        title=prefix+f"_variance {experiment_name}",
        graph_name=prefix+"_variance",
        lines=deepcopy(lines),
        groups=deepcopy(groups),
        should_flatten_graph=True,
        should_average=False,
        averaging_function=None,
        y_axis_name=prefix,
    )
    create_graph(
        title=prefix+f"_median {experiment_name}",
        graph_name=prefix+"_median",
        lines=deepcopy(lines),
        groups=deepcopy(groups),
        should_flatten_graph=True,
        should_average=True,
        averaging_function=median,
        y_axis_name=prefix,
    )
    create_graph(
        title=prefix+f"_mean {experiment_name}",
        graph_name=prefix+"_mean",
        lines=deepcopy(lines),
        groups=deepcopy(groups),
        should_flatten_graph=True,
        should_average=True,
        averaging_function=mean,
        y_axis_name=prefix,
    )

reward_lines, reward_groups = extract_accumulated_reward_as_lines(groups)
graph_variance_median_mean(
    lines=reward_lines,
    groups=reward_groups,
    prefix="reward",
)

curve_fit_lines, curve_fit_groups = extract_curve_fit_as_lines(groups)
graph_variance_median_mean(
    lines=curve_fit_lines,
    groups=curve_fit_groups,
    prefix="curve_loss",
)