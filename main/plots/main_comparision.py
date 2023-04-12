import ez_yaml
from blissful_basics import FS
import pandas as pd
import os

from config import config, path_to
from specific_tools.data_gathering import get_recorder_data
from generic_tools.plotting import graph_lines, xd_theme

lines = []
for name, data in get_recorder_data("@NO_ADJUSTER", "@NORMAL_ADJUSTER", "@PERFECT_ADJUSTER",):
    print(f'''processing {name}''')
    # data.records[0] = {"accumulated_reward": 0, "reward": 0, "timestep": 700, "line_fit_score": -0.18230862363710315}
    group_name = name.replace("@","").split("|")[0]
    reward_data = [ each for each in data.records if each.get("accumulated_reward", None) != None ]
    lines.append(
        dict(
            x_values=[ each.timestep           for each in reward_data ],
            y_values=[ each.accumulated_reward for each in reward_data ],
            name=name.replace("@","").replace("|"," ").lower(),
            color=dict(
                NO_ADJUSTER=xd_theme.red,
                NORMAL_ADJUSTER=xd_theme.blue,
                PERFECT_ADJUSTER=xd_theme.green,
            )[group_name],
        )
    )

# 
# flatten
# 
should_flatten_graph = True
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
should_average = True
if should_average:
    group_lines = {
        "no_adjuster": [],
        "normal_adjuster": [],
        "perfect_adjuster": [],
    }
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
                    propotion = relative_amount/the_range
                    return shift_towards(new_value=each_x, old_value=prev_x, propotion=propotion)
                
                prev_x = each_x
                prev_y = each_y
            
            # if its a vertical line or only has one point, this line will run
            return prev_y
                    
        return new_function
            
    for group_name, each_group_lines in group_lines.items():
        for each_line in lines:
            if group_name in each_line["name"]:
                each_group_lines.append(each_line)

    from statistics import median, mean as average
    new_lines = []
    for group_name, each_group_lines in group_lines.items():
        print(f'''computing average for {group_name}''')
        x_values = each_group_lines[0]["x_values"]
        functions = [ points_to_function(each["x_values"], each["y_values"]) for each in each_group_lines ]
        y_values = [
            median([ each_function(each_x) for each_function in functions ])
                for each_x in x_values
        ]
        new_lines.append(
            dict(
                x_values=x_values,
                y_values=y_values,
                name=group_name,
                color=dict(
                    no_adjuster=xd_theme.red,
                    normal_adjuster=xd_theme.blue,
                    perfect_adjuster=xd_theme.green,
                )[group_name],
            )
        )
    
    lines = new_lines

graph_lines(
    *lines,
    title="Heavy Noise Comparision",
    x_axis_name="Timestep",
    y_axis_name="AccumulatedReward",
    save_to="./main/plots/"+FS.name(__file__)+".html",
)