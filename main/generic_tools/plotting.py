from blissful_basics import wrap_around_get, stringify, FS

class Colors:
    def __init__(self, color_mapping):
        self._color_mapping = color_mapping
        for each_key, each_value in color_mapping.items():
            if isinstance(each_key, str) and len(each_key) > 0 and each_key[0] != '_':
                setattr(self, each_key, each_value)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return wrap_around_get(key, list(self._color_mapping.values()))
        elif isinstance(key, str):
            return self._color_mapping.get(key, None)
    
    def __repr__(self):
        return stringify(self._color_mapping)
    
    def __iter__(self):
        for each in self._color_mapping.values():
            yield each

xd_theme = Colors({
    "black":            '#000000',
    "white":            '#ffffff',
    "light_gray":       '#c7cbcd',
    "cement":           '#698098',
    "gray":             '#546e7a',
    "brown":            '#ce9178',
    "rust":             '#c17e70',
    "orange":           '#f78c6c',
    "yellow":           '#fec355',
    "bananna_yellow":   '#ddd790',
    "lime":             '#c3e88d',
    "green":            '#4ec9b0',
    "bold_green":       '#4ec9b0d0',
    "vibrant_green":    '#04d895',
    "dim_green":        '#80cbc4',
    "light_slate":      '#64bac5',
    "dark_slate":       '#3f848d',
    "light_blue":       '#89ddff',
    "blue":             '#82aaff',
    "electric_blue":    '#00aeffe7',
    "purple":           '#c792ea',
    "pink":             '#e57eb3',
    "red":              '#ff5572',
    "soft_red":         '#f07178',
})

def graph_lines(*args, title, x_axis_name, y_axis_name, save_to=None):
    """
        Example:
            graph_lines(
                dict(
                    x_values=[0,1,2,3],
                    y_values=[0,1,1,2],
                    name="line 1",
                    color="", # optional
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
    import plotly.express as px
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
    for line_index, line_info in enumerate(args):
        if line_info.get("color", None):
            fig.data[line_index].line.color = line_info["color"]
    if save_to:
        FS.ensure_is_folder(FS.parent_path(save_to))
        fig.write_html(save_to)
    fig.show()