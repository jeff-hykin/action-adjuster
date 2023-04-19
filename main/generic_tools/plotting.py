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
default_theme = xd_theme

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
        lines += each.lines
    
    # 
    # group average
    # 
    if callable(group_averaging_function):
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
        for group_index, (group_name, each_group) in enumerate(groups.items()):
            x_values = each_group["lines"][0]["x_values"]
                
            functions = [ points_to_function(each["x_values"], each["y_values"]) for each in each_group["lines"] ]
            y_values = [
                group_averaging_function([ each_function(each_x) for each_function in functions ])
                    for each_x in x_values
            ]
            new_lines.append(
                dict(
                    x_values=x_values,
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

