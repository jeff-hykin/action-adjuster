import ez_yaml
from blissful_basics import FS
import pandas as pd

from config import config, path_to

def graph_lines(*args, title, x_axis_name, y_axis_name, ):
    """
        Example:
            graph_lines(
                dict(
                    x_values=[0,1,2,3],
                    y_values=[0,1,1,2],
                    name="line 1"
                ),
                dict(
                    x_values=[0,1,2,3],
                    y_values=[0,1,1,2],
                    name="line 2"
                ),
                title="Linear vs. Non-Linear Energy Method",
                x_axis_name="X",
                y_axis_name="Displacement",
            )
    """
    import pandas as pd
    x_values = []
    y_values = []
    names = []
    for each in args:
        x_values += list(each["x_values"])
        y_values += list(each["y_values"])
        names += [each["name"]]*len(each["x_values"])
    data = {
        x_axis_name: x_values,
        y_axis_name: y_values,
        "Name": names,
    }
    df = pd.DataFrame(data)
    fig = px.line(df, x=x_axis_name, y=y_axis_name, color="Name", title=title)
    fig.show()

lines = []
for path in FS.list_paths_in(path_to.records):
    data = ez_yaml.to_object(string=FS.read(path+"/recorder.yaml"))
    records = data["records"]
    lines.append(
        dict(
            x_values=[ records ],
            y_values=y,
            name=FS.basename(path),
        ),
    )