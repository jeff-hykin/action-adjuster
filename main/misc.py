def pascal_case_with_spaces(string):
    string = f"{string}"
    digits = "1234567890-"
    valid_word_contents = "1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM-"
    new_string = " "
    # get pairwise elements
    for each_character in string:
        prev_character = new_string[-1]
        prev_is_lowercase = prev_character.lower() == prev_character
        each_is_uppercase = each_character.lower() != each_character
        
        # remove misc characters (handles snake case, kebab case, etc)
        if each_character not in valid_word_contents:
            new_string += " "
        # start of word
        elif prev_character not in valid_word_contents:
            new_string += each_character.upper()
        # start of number
        elif prev_character not in digits and each_character in digits:
            new_string += each_character
        # end of number
        elif prev_character in digits and each_character not in digits:
            new_string += each_character.upper()
        # camel case
        elif prev_is_lowercase and each_is_uppercase:
            new_string += " "+each_character.upper()
        else:
            new_string += each_character
    
    # flatten out all the whitespace
    new_string = new_string.strip()
    while "  " in new_string:
        new_string = new_string.replace("  "," ")
    
    return new_string


def no_duplicates(items): # preserving order
    copy = []
    for each in items:
        if each in copy:
            continue
        copy.append(each)
    return copy

def line_count_of(file_path):
    # from stack overflow "how to get a line count of a large file cheaply"
    def _make_gen(reader):
        while 1:
            b = reader(2**16)
            if not b: break
            yield b
    with open(file_path, "rb") as file:
        count = sum(buf.count(b"\n") for buf in _make_gen(file.raw.read))
    
    return count

def is_valid_decimal_number(input_str):
    import re
    # Define a regular expression pattern to match positive or negative decimal numbers
    pattern = r'^[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?$'
    # Use the re.match function to check if the input string matches the pattern
    if re.match(pattern, input_str):
        return True
    else:
        return False
            
import pytz
def excel_datetime_to_unixtime_converter(timezone="America/Chicago", human_format="%d-%b-%y %H:%M:%S"):
    """
    Example:
        converter = excel_datetime_to_unixtime_converter(
            timezone="America/Chicago",
            human_format="%d-%b-%y %H:%M:%S",
        )
        converter("11-Aug-23 11:00:00")
        converter(45148.9583333333)
    """
    import pytz
    from datetime import datetime, timedelta

    local_timezone = pytz.timezone(timezone)
    human_datetime_to_unix_datetime = lambda a_datetime: local_timezone.localize(datetime.strptime(a_datetime, human_format)).timestamp()

    def excel_serial_to_datetime(excel_date_value):
        # Determine the base date for Excel (Windows)
        base_date = datetime(1899, 12, 30)
        if excel_date_value != excel_date_value:
            return excel_date_value # NaN
        # Calculate the number of days from the base date
        delta_days = timedelta(days=excel_date_value)
        # Calculate the final datetime value
        result_datetime = base_date + delta_days
        return result_datetime
            
    import numbers
    def convert(value):
        if isinstance(value, numbers.Number) or is_valid_decimal_number(value):
            return excel_serial_to_datetime( float(value) ).timestamp()
        elif isinstance(value, str):
            return human_datetime_to_unix_datetime(value)
    
    return convert

import csv
import os

def get_columns(input_csv_file):
    with open(input_csv_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        return next(csv_reader)

def make_headers_unique(header, rename_warning=False):
    theres_a_duplicate = len(set(header)) != len(header)
    if not theres_a_duplicate:
        return header
    
    working_header = list(header)
    while theres_a_duplicate:
        def remove_number_base(string):
            if "_" in string:
                *parts, last = string.split("_")
                if last.isdigit():
                    return string[0:-(len(last)+1)]
            return string
        
        duplicates = []
        duplicate_checker = []
        for each_column_name in working_header:
            if each_column_name in duplicate_checker:
                duplicates.append(each_column_name)
            else:
                duplicate_checker.append(each_column_name)
        
        basename_duplicates = tuple(set(remove_number_base(each) for each in duplicates))
        new_header_names = []
        basename_counts = { key: 0 for key in basename_duplicates }
        for each_column_name in working_header:
            basename = remove_number_base(each_column_name)
            if basename in basename_duplicates:
                basename_counts[basename] += 1
                new_name = f'{basename}_{basename_counts[basename]}'
                new_header_names.append(new_name)
            else:
                new_header_names.append(each_column_name)
        
        working_header = new_header_names
        theres_a_duplicate = len(set(working_header)) != len(working_header)
    
    rename_warning and print("Needed to rename some column names to avoid duplicates")
    for each_old_name, each_new_name in zip(header, working_header):
        if each_old_name != each_new_name:
            rename_warning and print(f"    {repr(each_old_name)} => {repr(each_new_name)}")
    
    return working_header
 
# Function to split a CSV file into separate CSV files, one per column
def split_csv_into_column_files(input_csv_file, output_directory, name_changer=None, new_column_prefix="", exclude_columns=[], everywhere_columns=[], simplify_column_names=True, show_warnings=True):
    everywhere_columns = tuple(everywhere_columns)
    exclude_columns    = tuple(exclude_columns)
    # Create a directory to store the output CSV files
    os.makedirs(output_directory, exist_ok=True)
    
    transform_name = lambda each: (new_column_prefix if each not in everywhere_columns else "")+each.replace("/","_") # slashes cannot be in path names
    if simplify_column_names:
        transform_name = lambda each: (new_column_prefix if each not in everywhere_columns else "")+pascal_case_with_spaces(each.strip()).replace(" ", "_").lower()
    
    if name_changer == None:
        name_changer = lambda each: each
    
    # Read the input CSV file
    with open(input_csv_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        original_header = next(csv_reader)  # Read the header row
        
        index_of_everywhere_columns = {}
        for each in everywhere_columns:
            if each not in original_header:
                if show_warnings:
                    print(f"when calling split_csv_into_column_files()\n   {repr(each)} is one of the everywhere_columns, but its not in the actual column list: {original_header}")
                continue
            else:
                index_of_everywhere_columns[each] = original_header.index(each)
        
        index_of_exclude_columns = {}
        for each in exclude_columns:
            if each not in original_header:
                if show_warnings:
                    print(f"when calling split_csv_into_column_files()\n   {repr(each)} is one of the exclude_columns, but its not in the actual column list: {original_header}")
                continue
            else:
                index_of_exclude_columns[each] = original_header.index(each)
                
        # simplify the names if needed
        header = [ name_changer(transform_name(each)) for each in original_header ]
        header = make_headers_unique(header, rename_warning=show_warnings)
        everywhere_columns = [ header[each_index] for each_index in index_of_everywhere_columns.values() ]
        exclude_columns    = [ header[each_index] for each_index in index_of_exclude_columns.values()    ]
        
        # Create an output CSV file for each column
        column_files = {}
        try:
            for column_name in header:
                if column_name in everywhere_columns or column_name in exclude_columns:
                    continue
                column_filename = os.path.join(output_directory, f'{column_name}.csv')
                column_files[column_name] = open(column_filename, 'w', newline='')

            # Write the headers to each column file
            for column_name, column_file in column_files.items():
                csv.writer(column_file).writerow([*everywhere_columns,column_name])

            # Split the data into columns
            for row in csv_reader:
                row_as_dict = { key: value for key, value in zip(header,row)}
                for index, column_value in enumerate(row):
                    column_name = header[index]
                    if column_name not in everywhere_columns and column_name not in exclude_columns:
                        column_file = column_files[column_name]
                        csv.writer(column_file).writerow([ *[row_as_dict[each] for each in everywhere_columns], column_value])
        finally:
            # Close all column files
            for column_file in column_files.values():
                try:
                    column_file.close()
                except Exception as error:
                    pass


def simplify_names(names, show_warnings=True):
    transform_name = lambda each: pascal_case_with_spaces(each.strip()).replace(" ", "_").lower()
            
    # simplify the names if needed
    names = [ transform_name(each) for each in names ]
    return make_headers_unique(names, rename_warning=show_warnings)

def prev_next(iterable):
    iterator = iter(iterable)
    each = next(iterator)
    while True:
        prev = each
        each = next(iterator)
        yield prev, each

def hours_to_seconds(hours):
    return hours*60*60

def seconds_to_hours(seconds):
    return seconds/60/60
    
def unix_timestamp_to_human(timestamp):
    import datetime
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def compute_r_correlation(x_values, y_values):
    from sklearn.linear_model import LinearRegression
    regression = LinearRegression()
    regression.fit(x_values.reshape(-1, 1), y_values)
    r_squared_for_mean = regression.score(x_values.reshape(-1, 1), y_values)
    
    return r_squared_for_mean

from statistics import median
def linear_regression(x, y):
    number_of_values = len(y)
    assert len(x) == number_of_values, "need to have same number of x and y points"
    # Calculate the mean of x and y
    mean_x = sum(x) / number_of_values
    mean_y = sum(y) / number_of_values

    # Calculate the slope (m) and intercept (b)
    numerator = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(number_of_values)])
    denominator = sum([(x[i] - mean_x) ** 2 for i in range(number_of_values)])
    
    if denominator == 0:
        return None, None
    
    slope = numerator / denominator
    offset = mean_y - slope * mean_x

    return slope, offset

def basic_interpolate(x0,y0,x1,y1,x):
    # verticle line
    if x0 == x1:
        # assume consistent rate of change in y (which may be 0)
        return y0 + (y0 - y1)
    
    slope = (y1 - y0) / (x1 - x0)
    y = y0 + slope * (x - x0)
    return y

from statistics import quantiles
import numbers
nan = float("NaN")
def interpolate(x_values, y_values, *, are_sorted=False, close_enough=None, max_gap=lambda gaps: quantiles(gaps)[2], extrapolate=True):
    """
    Example:
        x = [1, 2, 3, 10,11,12]
        y = [2, 3, 6,  7, 8,10]
        func = interpolate(x, y, max_gap=1) 
        func(0.5) == 1.5
        func(1.5) == 2.5
        func(3.0) == 6.0
        func(3.5) == 7.5
        func(3.6) == float("NaN")
        func(4.0) == float("NaN")
        func(9.0) == float("NaN")
        func(9.5) == 6.5
        func(10.0) == 7.0
        func(12.0) == 10.0
        func(12.5) == 11.0
        func(12.6) == float("NaN")
    Args:
        max_gap:
            - Defaults to the 75th percentile of all gaps (which would also be 1 in this case)
            - Can be a function (which will be given an argument of all the gap-lengths)
            - Can equal None in order to force interpolation (extrapolation)
    """
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
                ((x,y) for x,y in zip(x_values, y_values) if x!=None and y!=None),
                key=lambda each: each[0],
            )
        )
    
    if callable(max_gap):
        max_gap = max_gap(tuple(each-prev for prev, each in zip(x_values[0:-1], x_values[1:])))
    
    if max_gap != None:
        max_gap = max_gap/2
    
    minimum_x = x_values[0]
    maximum_x = x_values[-2] # not the true max, but, because of indexing, the 2nd-maximum
    
    if close_enough == None:
        handle_close_enough = lambda *args: nan
    else:
        def handle_close_enough(x0,x1,x):
            # beyond lower limit
            if x0-close_enough > x:
                return nan
            # beyond upper limit
            if x1+close_enough < x:
                return nan
            # close enough to x0
            if x0+close_enough > x:
                return x0
            # close enough to x0
            if x1-close_enough < x:
                return x1
    
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
            
            if low > 0 and x <= x_values[low]:
                low -= 1
            
            x_index = low
        
        
        # Perform linear interpolation / extrapolation
        x0, x1 = x_values[x_index], x_values[x_index+1]
        y0, y1 = y_values[x_index], y_values[x_index+1]
        
        if max_gap == None:
            return basic_interpolate(x0,y0,x1,y1,x)
        
        is_outside_range = x > x1 or x < x0
        if not extrapolate and is_outside_range:
            return nan
        close_enough_to_lower = abs(x0-x) <= max_gap
        close_enough_to_upper = abs(x1-x) <= max_gap
        segment_break = (x1 - x0) > (max_gap*2) # interpolation points are too far apart to justify an interpolation
        
        if is_outside_range:
            if not close_enough_to_lower and not close_enough_to_upper:
                return handle_close_enough(x0,x1,x)
            elif segment_break:
                return handle_close_enough(x0,x1,x)
            else:
                return basic_interpolate(x0,y0,x1,y1,x)
        # if x is inbewteen the points
        else:
            if not segment_break:
                return basic_interpolate(x0,y0,x1,y1,x)
            else:
                if close_enough_to_upper:
                    if x1-x == 0:
                        return y1
                    another_upper_point_exists = x_index+2 < len(x_values)
                    if another_upper_point_exists:
                        # interpolation points are too far apart to justify an interpolation
                        another_upper_point_can_be_interpolated = (x_values[x_index+2] - x_values[x_index+1]) <= (max_gap*2)
                        if another_upper_point_can_be_interpolated:
                            return basic_interpolate(
                                x_values[x_index+1],
                                y_values[x_index+1],
                                x_values[x_index+2],
                                y_values[x_index+2],
                                x
                            )
                    # return float("NaN")
                elif close_enough_to_lower:
                    if x0-x == 0:
                        return y0
                    another_lower_point_exists = x_index-1 > 0
                    if another_lower_point_exists:
                        # interpolation points are too far apart to justify an interpolation
                        another_lower_point_can_be_interpolated = (x_values[x_index] - x_values[x_index-1]) <= (max_gap*2)
                        if another_lower_point_can_be_interpolated:
                            return basic_interpolate(
                                x_values[x_index-1],
                                y_values[x_index-1],
                                x_values[x_index],
                                y_values[x_index],
                                x
                            )
                            
                    # return float("NaN")
                # else:
                #     return float("NaN")
        return handle_close_enough(x0,x1,x)
    
    inner_function.max_gap = max_gap
    return inner_function

def sliding_window(iterable, window_size):
    window = []
    for each in iterable:
        window.append(each)
        window = window[-window_size:]
        if len(window) == window_size:
            yield window


import math
import numpy
from statistics import median
def outlier_smoothing(data, window_size=7, stdev_limit=4, include_tails=False):
    outputs = numpy.zeros(len(data))
    window = []
    assert window_size > 4
    if len(data) > window_size:
        return data
    if not isinstance(data, numpy.ndarray):
        data = numpy.array(data)
    
    x = numpy.array(tuple(range(window_size)))
    mean_x = x.mean()
    precaled_1 = (x - mean_x)
    denominator = (precaled_1 ** 2).sum()
    last_index = len(data)-window_size
    for index in range(last_index+1):
        y = numpy.array(data[index:index+window_size])
        mean_y = y.mean()
        numerator = (precaled_1 * (y - mean_y)).sum()
        slope = numerator / denominator
        offset = mean_y - slope * mean_x
        
        distances = numpy.abs((slope*x - y) + offset) / (math.sqrt(slope**2 + 1))
        distances = (distances - distances.min())**2
        stdevs = []
        for each_index in range(len(distances)):
            indicies = numpy.ones(len(distances))==1
            indicies[each_index] = False
            stdevs.append(
                distances[indicies].std()
            )
        print(f'''distances = {distances}''')
        standard_deviation = min(stdevs)
        print(f'''standard_deviation = {standard_deviation}''')
        print(f'''standard_deviation*stdev_limit = {standard_deviation*stdev_limit}''')
        upper_bound = median(distances) + (standard_deviation*stdev_limit)
        print(f'''upper_bound = {upper_bound}''')
        good_indicies = (distances <= upper_bound)
        
        y_ = y[good_indicies]
        x_ = x[good_indicies]
        mean_y = y_.mean()
        
        mean_x_ = x_.mean()
        precaled_1_ = (x_ - mean_x_)
        denominator_ = (precaled_1_ ** 2).sum()
        
        numerator = (precaled_1_ * (y_ - mean_y)).sum()
        
        slope = numerator / denominator
        offset = mean_y - slope * mean_x_
        bad_indicies = numpy.logical_not(good_indicies)
        print(f'''y[bad_indicies] = {y[bad_indicies]}''')
        
        y[bad_indicies] = (slope*x + offset)[bad_indicies]
        if any(bad_indicies[1:-1]):
            for bad_index, each in enumerate(bad_indicies[:-1]):
                if each and bad_index > 1:
                    y[bad_index] = (data[index:index+window_size][bad_index+1]+data[index:index+window_size][bad_index-1])/2
        
        if index == 0:
            if include_tails:
                outputs[0:window_size//2] = y[0:window_size//2]
            else:
                outputs[0:window_size//2] = data[index:index+window_size][0:window_size//2]
        if index == last_index:
            if include_tails:
                outputs[index+(window_size//2):] = y[(window_size//2):]
            else:
                outputs[index+(window_size//2):] = data[index:index+window_size][(window_size//2):]
        
        # append the middle value
        outputs[window_size//2+index] = y[window_size//2]
    
    return outputs

class Duration(float):
    def __new__(cls, days=None, hours=None, minutes=None, seconds=None, milliseconds=None, unit_floor=None, start_value=None):
        if unit_floor==None:
            if milliseconds!=None:
                unit_floor = 1
            elif seconds!=None:
                unit_floor = 1000
            elif minutes!=None:
                unit_floor = 1000*60
            elif hours!=None:
                unit_floor = 1000*60*60
            elif days!=None:
                unit_floor = 1000*60*60*24
            else:
                raise Exception(f'''Duration() was called without an argument (if you want a duration of 0, then do Duration(seconds=0))''')
        
        start_value = start_value or (
            (days or 0)/(86_400_000/unit_floor) +
            (hours or 0)/(3_600_000/unit_floor) +
            (minutes or 0)/(60_000/unit_floor) +
            (seconds or 0)/(1000/unit_floor) +
            (milliseconds or 0)*(unit_floor)
        )
        output = super().__new__(cls, start_value)
        output.unit_floor = unit_floor
        return output
    
    @property
    def days(self): return self/(86_400_000/self.unit_floor)
    @property
    def hours(self): return self/(3_600_000/self.unit_floor)
    @property
    def minutes(self): return self/(60_000/self.unit_floor)
    @property
    def seconds(self): return self/(1000/self.unit_floor)
    @property
    def milliseconds(self): return self*(self.unit_floor)
    
    def __add__(self, other):
        if isinstance(other, Duration):
            return Duration(
                start_value=((0+other) + (0+self)),
                unit_floor=self.unit_floor,
            )
        elif isinstance(other, numbers.Number):
            return Duration(
                start_value=(0+self)+other,
                unit_floor=self.unit_floor
            )
    
    def __hash__(self):
        return f"Duration(minutes={self.minutes})"

def recursive_update(old_values, new_values):
    if not isinstance(old_values, dict) or not isinstance(new_values, dict):
        raise TypeError('Params of recursive_update should be dicts')

    for key in new_values:
        if isinstance(new_values[key], dict) and isinstance(old_values.get(key), dict):
            old_values[key] = recursive_update(old_values[key], new_values[key])
        else:
            old_values[key] = new_values[key]

    return old_values

def class_to_dict(a_class=None, *, recursive=False, override_with={}):
    if a_class == None:
        # assume this is a decorator call (just a way of passing args)
        return lambda a_class: class_to_dict(a_class, recursive=recursive, override_with=override_with)
    # dectorator no args, or normal function call
    else:
        from blissful_basics import LazyDict
        output = LazyDict()
        if isinstance(a_class, dict):
            output = LazyDict(a_class)
        else:
            for each in dir(a_class):
                if each[0] != "_":
                    try:
                        value = getattr(a_class, each)
                        # if
                        if recursive and isinstance(value, type):
                            output[each] = class_to_dict(value)
                        else:
                            output[each] = value
                    except Exception as error:
                        pass
        
        if isinstance(override_with, type):
            override_with = class_to_dict(override_with, recursive=recursive)
        
        recursive_update(old_values=output, new_values=override_with)
        return output

def pairwise_remove_nan(*values):
    combined = zip(*values)
    iterable_thing = zip(*(each for each in combined if all(each_inner==each_inner for each_inner in each)))
    iterable = iter(iterable_thing)
    try:
        first_output = next(iterable)
        yield first_output
        yield from iterable
    except Exception as error:
        for each in range(len(values)):
            yield []

def pairwise_remove(*values, any_of=math.nan):
    iterable_thing = zip(*values)
    # nan is edgecase
    if any(each!=each for each in any_of):
        iterable_thing = (each for each in iterable_thing if all(each_inner==each_inner for each_inner in each))
    # normal case
    iterable_thing = (each for each in iterable_thing if all((each_inner not in any_of) for each_inner in each))
    # unzip
    iterable = iter(zip(*iterable_thing))
    try:
        first_output = next(iterable)
        yield first_output
        yield from iterable
    except Exception as error:
        for each in range(len(values)):
            yield []
 

# 
# StorageDict
# 
if True:
    import warnings
    from pathlib import Path
    try:
        
        from ez_yaml import get_yaml
        from __dependencies__.ez_yaml import get_yaml
        _open_files = []
        _yaml_obj_for_bound_dict = get_yaml(dict(
            safe=False,
            width=None,
            allow_duplicate_keys="overwrite", # required for StorageDict
            explicit_start=False,
            explicit_end=False,
            explict_null=True,
            indent_mapping=3,
            indent_sequence=2,
            offset=0,
        ))
        def indent(string, by):
            indent_string = (" "*by)
            return indent_string + string.replace("\n", "\n"+indent_string)

        def stringify(value, onelineify_threshold=None):
            if onelineify_threshold is None: onelineify_threshold = stringify.onelineify_threshold
            
            length = 0
            if isinstance(value, str):
                return repr(value)
            elif isinstance(value, dict):
                if len(value) == 0:
                    return "{}"
                else:
                    # if all string keys and all identifiers
                    if all(isinstance(each, str) and each.isidentifier() for each in value.keys()):
                        items = value.items()
                        output = "dict(\n"
                        for each_key, each_value in items:
                            element_string = each_key + "=" + stringify(each_value)
                            length += len(element_string)+2
                            output += indent(element_string, by=4) + ", \n"
                        output += ")"
                        if length < onelineify_threshold:
                            output = output.replace("\n    ","").replace("\n","")
                        return output
                    # more complicated mapping
                    else:
                        items = value.items()
                        output = "{\n"
                        for each_key, each_value in items:
                            element_string = stringify(each_key) + ": " + stringify(each_value)
                            length += len(element_string)+2
                            output += indent(element_string, by=4) + ", \n"
                        output += "}"
                        if length < onelineify_threshold:
                            output = output.replace("\n    ","").replace("\n","")
                        return output
            elif isinstance(value, list):
                if len(value) == 0:
                    return "[]"
                output = "[\n"
                for each_value in value:
                    element_string = stringify(each_value)
                    length += len(element_string)+2
                    output += indent(element_string, by=4) + ", \n"
                output += "]"
                if length < onelineify_threshold:
                    output = output.replace("\n    ","").replace("\n","")
                return output
            elif isinstance(value, set):
                if len(value) == 0:
                    return "set([])"
                output = "set([\n"
                for each_value in value:
                    element_string = stringify(each_value)
                    length += len(element_string)+2
                    output += indent(element_string, by=4) + ", \n"
                output += "])"
                if length < onelineify_threshold:
                    output = output.replace("\n    ","").replace("\n","")
                return output
            elif isinstance(value, tuple):
                if len(value) == 0:
                    return "tuple()"
                output = "(\n"
                for each_value in value:
                    element_string = stringify(each_value)
                    length += len(element_string)+2
                    output += indent(element_string, by=4) + ", \n"
                output += ")"
                if length < onelineify_threshold:
                    output = output.replace("\n    ","").replace("\n","")
                return output
            else:
                try:
                    debug_string = value.__repr__()
                except Exception as error:
                    from io import StringIO
                    import builtins
                    string_stream = StringIO()
                    builtins.print(value, file=string_stream)
                    debug_string = string_stream.getvalue()
                
                # TODO: handle "<slot wrapper '__repr__' of 'object' objects>"
                if debug_string.startswith("<class") and hasattr(value, "__name__"):
                    return value.__name__
                if debug_string.startswith("<function <lambda>"):
                    return "(lambda)"
                if debug_string.startswith("<function") and hasattr(value, "__name__"):
                    return value.__name__
                if debug_string.startswith("<module") and hasattr(value, "__name__"):
                    _, *file_path, _, _ = debug_string.split(" ")[-1]
                    file_path = "".join(file_path)
                    return f"module(name='{value.__name__}', path='{file_path}')"
                
                space_split = debug_string.split(" ")
                if len(space_split) >= 4 and debug_string[0] == "<" and debug_string[-1] == ">":
                    
                    if space_split[-1].startswith("0x") and space_split[-1] == "at":
                        _, *name_pieces = space_split[0]
                        *parts, name = "".join(name_pieces).split(".")
                        parts_str = ".".join(parts)
                        return f'{name}(from="{parts_str}")'
                
                return debug_string
        stringify.onelineify_threshold = 50
        class StorageDictCantWrite(Warning):
            pass
        _storage_dict_info = {}
        class StorageDict(dict):
            """
            Example:
                # all keys will be real time saved/loaded to/from a file
                # if the file doesn't exist it will be created
                StorageDict(dict(thing="default value"), path="file_path.yaml")
            """
            class Path: pass
            class File: pass
            yaml = _yaml_obj_for_bound_dict
            def __init__(self, *args, path):
                data = None
                try:
                    as_path_object = Path(path)
                    print(f'''loading file''')
                    input("press enter")
                    data = StorageDict.yaml.load(as_path_object)
                    print(f'''loaded''')
                    
                    # compress any duplicate keys (which is going to be extremely common)
                    print(f'''removing dup keys''')
                    input("press enter")
                    with as_path_object.open('w') as output_file:
                        StorageDict.yaml.dump(data, output_file)
                    print(f'''done''')
                except:
                    pass
                
                if not isinstance(data, dict):
                    print(f'''not isinstance(data, dict), data: {data}''')
                    data = {}
                    # ensure parent folder exists
                    import os
                    directory = os.path.dirname(path)
                    if directory == '': directory = '.'
                    os.makedirs(directory, exist_ok=True)
                    print(f'''clearing file''')
                    input("press enter")
                    with open(path, 'w+') as the_file:
                        the_file.write("")
                
                file = open(path, 'a+')
                _storage_dict_info[id(self)] = (path, file)
                _open_files.append(file)
                default = dict(*args)
                default.update(data)
                # default vaue
                super(StorageDict, self).__init__(default)
                self.__dict__ = self
            
            def __getitem__(self, key):
                if key == StorageDict.Path: return _storage_dict_info[id(self)][0]
                if key == StorageDict.File: return _storage_dict_info[id(self)][1]
                return self.__dict__.get(key)
            
            def __setitem__(self, key, value):
                path, file = _storage_dict_info[id(self)]
                new_entry = {key:value}
                try:
                    StorageDict.yaml.dump(new_entry, file)
                except Exception as error:
                    warnings.warn(StorageDictCantWrite(f"Unable to set key for {path}: {error}\ntype(key)={type(key)}\ntype(value)={type(value)}"))
                super(StorageDict, self).update(new_entry)
                
            def __delitem__(self, key):
                try:
                    del self.__dict__[key]
                    _, file = _storage_dict_info[id(self)]
                    # if something was removed, refresh the whole file
                    file.truncate(0)
                    StorageDict.yaml.dump(self.__dict__, file)
                    # TODO: theres a more efficient way to delete, but its harder (yaml-tag a null value as a delete operation, and do custom handling)
                except Exception as error:
                    pass
            
            def __str__(self):
                if len(self.__dict__) == 0:
                    return "{}"
                return stringify(self.__dict__)
            
            def __repr__(self):
                return self.__str__()
            
            def merge(self, other_dict=None, **kwargs):
                other_dict = dict(other_dict or {})
                other_dict.update(kwargs)
                a_copy = dict(self)
                a_copy.update(other_dict)
                return a_copy
            
            def update(self, other_dict):
                for each_key, each_value in other_dict.items():
                    self[each_key] = each_value
                return self
            
            def setdefault(self, *args, **kwargs):
                return super(StorageDict, self).setdefault(*args, **kwargs)
            
            def __copy__(self):
                return StorageDict(self.__dict__)
            
            def __deepcopy__(self, memo):
                from copy import deepcopy
                
                address = id(self)
                if address in memo:
                    return memo[address]
                new = memo[address] = StorageDict(self.__dict__)
                for each_key, each_value in new.items():
                    new[each_key] = deepcopy(each_value, memo)
                return new

        # cleanup
        import atexit
        @atexit.register
        def _():
            for each in _open_files:
                try:
                    each.close()
                except Exception as error:
                    pass
    except Exception as error:
        print(f"Error when setting up StorageDict: {error}")

class ObjectInspector:
    def __init__(self, obj):
        self.obj = obj
        self.visited = set()

    def get_attributes(self, obj=None, prefix=""):
        if obj is None:
            obj = self.obj

        attributes = {}

        # Avoid infinite recursion by tracking visited objects
        if type(obj) == None:
            return {}
        if id(obj) in self.visited:
            return {}

        self.visited.add(id(obj))

        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in obj.__dict__.items():
                if attr_name.startswith("__"):
                    continue
                attribute_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
                attributes[attribute_prefix] = self.get_attributes(attr_value, attribute_prefix)

        # Include methods
        for method_name in dir(obj):
            if callable(getattr(obj, method_name, None)) and not method_name.startswith("__"):
                method_prefix = f"{prefix}.{method_name}" if prefix else method_name
                attributes[method_prefix] = "method"

        return attributes

def module_interface_to_json(module):
    import json
    return json.dumps(ObjectInspector(module).get_attributes())

def are_equal(a, b):
    try:
        return True if a == b else False
    except Exception as error:
        # deals with the stupid "The truth value of an array with more than one element is ambiguous" problem
        if hasattr(a, '__iter__'):
            if hasattr(b, '__iter__'):
                if len(a) != len(b):
                    return False
                if isinstance(a, str):
                    if isinstance(b, str):
                        return True if a == b else False
                    else:
                        return False
                if isinstance(a, dict):
                    if isinstance(b, dict):
                        return are_equal(a.keys(),b.keys()) and are_equal(b.values(), a.values())
                    else:
                        return False
                try:
                    return True if hash(a) == hash(b) else False
                except Exception as error:
                    for i in range(len(a)):
                        value1 = a[i]
                        value2 = b[i]
                        theyre_equal = are_equal(value1, value2)
                        if not theyre_equal:
                            return False
                return True
            else:
                return False
        else:
            raise error

def scaled_sigmoid(x):
    # normally sigmoid(10) = 0.9999092042625952
    # normally sigmoid(100) = 1.0
    # this streches it out to be sigmoid(1000) = 0.4621171572600098
    x = x / 1000
    return ((1 / (1 + math.exp(-x))) - 0.5) * 2