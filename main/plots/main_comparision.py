import ez_yaml
from blissful_basics import FS
import pandas as pd

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

graph_lines(
    *lines,
    title="Heavy Noise Comparision",
    x_axis_name="Timestep",
    y_axis_name="AccumulatedReward",
    save_to=FS.local_path(FS.name(__file__)+".html"),
)