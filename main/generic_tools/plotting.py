from blissful_basics import wrap_around_get, stringify, FS, print

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
default_theme = xd_theme

def points_to_function(x_values, y_values, are_sorted=False):
    number_of_values = len(x_values)
    if number_of_values != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if number_of_values == 0:
        raise ValueError("called points_to_function() but provided an empty list of points")
    # horizontal line
    if number_of_values == 1:
        return lambda x_value: y_values[0]
    
    if not are_sorted:
        # sort to make sure x values are least to greatest
        x_values, y_values = zip(
            *sorted(
                zip(x_values, y_values),
                key=lambda each: each[0],
            )
        )
    
    minimum_x = x_values[0]
    maximum_x = x_values[-2] # not the true max, but, because of indexing, the 2nd-maximum
    def inner_function(x):
        if x >= maximum_x:
            # needs -2 because below will do x_values[x_index+1]
            x_index = number_of_values-2
        elif x <= minimum_x:
            x_index = 0
        else:
            # binary search for x
            low = 0
            high = number_of_values - 1

            while low < high:
                mid = (low + high) // 2

                if x_values[mid] < x:
                    low = mid + 1
                else:
                    high = mid

            if low > 0 and x < x_values[low - 1]:
                low -= 1
            
            x_index = low
        
        # Perform linear interpolation / extrapolation
        x0, x1 = x_values[x_index], x_values[x_index+1]
        y0, y1 = y_values[x_index], y_values[x_index+1]
        slope = (y1 - y0) / (x1 - x0)
        y = y0 + slope * (x - x0)

        return y
    
    return inner_function

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
    # print(title)
    # with print.indent:
    #     print(stringify(args))
    
    if len(args) == 0:
        raise Exception(f'''\n\ngraph_lines(\n    title={title},\n    x_axis_name={x_axis_name},\n    y_axis_name={y_axis_name}\n)\nwas called without any normal args (e.g. no lines/line-data given)''')
    
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

from copy import deepcopy
def graph_groups(
    groups,
    remove_space_below_individual=False,
    group_averaging_function=None,
    theme=default_theme,
    **kwargs,
):
    groups = deepcopy(groups)
    lines = []
    for each in groups.values():
        lines += each["lines"]
    
    # 
    # group average
    # 
    if callable(group_averaging_function):
        new_lines = []
        for group_index, (group_name, each_group) in enumerate(groups.items()):
            lines = each_group["lines"]
            
            functions = [
                points_to_function(
                    each["x_values"],
                    each["y_values"],
                )
                    for each in lines
            ]
            
            # x values might not be the same across lines, so get all of them
            any_x_value = set()
            for each_line in lines:
                any_x_value |= set(each_line["x_values"])
            any_x_value = sorted(any_x_value)
            
            y_values = [
                group_averaging_function([ each_function(each_x) for each_function in functions ])
                    for each_x in any_x_value
            ]
            new_lines.append(
                dict(
                    x_values=any_x_value,
                    y_values=y_values,
                    name=group_name,
                    color=each_group.get("color", theme[group_index]),
                )
            )
        
        lines = new_lines
    # 
    # flatten
    # 
    if remove_space_below_individual:
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
        **kwargs,
    )

