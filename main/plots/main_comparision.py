import ez_yaml
from blissful_basics import FS
import pandas as pd

from config import config, path_to
from specific_tools.data_gathering import get_recorder_data
from generic_tools.plotting import graph_lines

lines = []
for name, data in get_recorder_data("@NO_ADJUSTER|0", "@NORMAL_ADJUSTER|0", "@PERFECT_ADJUSTER|0",):
    # data = {"accumulated_reward": 0, "reward": 0, "timestep": 700, "line_fit_score": -0.18230862363710315}
    lines.append(
        dict(
            x_values=[ each.timestep           for each in data.records ],
            y_values=[ each.accumulated_reward for each in data.records ],
            name=name.replace("@","").replace("|"," ").lower(),
        )
    )

graph_lines(
    *lines,
    title="Line Fit Score",
    x_axis_name="Timestep",
    y_axis_name="AccumulatedReward",
)