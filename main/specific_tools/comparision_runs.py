import sys
import subprocess
import json
from config import path_to
from subprocess import Popen, PIPE

finished = 0
number_of_data_points_per_scenario = 30
for run_number in range(finished, number_of_data_points_per_scenario):
    processes = []
    for each_profile in [ "@NORMAL_ADJUSTER" ]: # "@NO_ADJUSTER", "@PERFECT_ADJUSTER",
        run_number_string = f"{run_number}".rjust(3,' ')
        output_folder = f"{path_to.records}_50/{each_profile}|{run_number_string}.ignore"
        print(f"working on: {output_folder}")
        process = Popen([ sys.executable, path_to.main, "@WARTHOG", "@HEAVY_NOISE", each_profile, f"output_folder:{json.dumps(output_folder)}",  ])
        processes.append(process)
    
    for each_process in processes:
        each_process.wait()