import ez_yaml
from blissful_basics import FS
import pandas as pd

from config import config, path_to
from specific_tools.data_gathering import get_recorder_data
from generic_tools.plotting import graph_lines

lines = []
for name, data in get_recorder_data("@NO_ADJUSTER|0.ignore", "@NORMAL_ADJUSTER|0.ignore", "@PERFECT_ADJUSTER|0.ignore",):
    print(f'''processing {name}''')
    # data.records[0] = {"accumulated_reward": 0, "reward": 0, "timestep": 700, "line_fit_score": -0.18230862363710315}
    reward_data = [ each for each in data.records if each.get("accumulated_reward", None) != None ]
    lines.append(
        dict(
            x_values=[ each.timestep           for each in reward_data ],
            y_values=[ each.accumulated_reward for each in reward_data ],
            name=name.replace("@","").replace("|"," ").lower(),
        )
    )

graph_lines(
    *lines,
    title="Line Fit Score",
    x_axis_name="Timestep",
    y_axis_name="AccumulatedReward",
)