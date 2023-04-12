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

graph_lines(
    *lines,
    title="Heavy Noise Comparision",
    x_axis_name="Timestep",
    y_axis_name="AccumulatedReward",
    save_to="./main/plots/"+FS.name(__file__)+".html",
)