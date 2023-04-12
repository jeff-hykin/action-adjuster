import sys
import subprocess
import json
from config import path_to
from subprocess import Popen, PIPE

number_of_iterations = 30
for run_number in range(30):
    for each_profile in [ "@NO_ADJUSTER", "@PERFECT_ADJUSTER", "@NORMAL_ADJUSTER" ]:
        output_folder = f"{path_to.records}/{each_profile}|{run_number}"
        print(f"working on: {output_folder}")
        proc = Popen([ sys.executable, path_to.main, "@WARTHOG", "@HEAVY_NOISE", each_profile, f"output_folder:{json.dumps(output_folder)}",  ])
        proc.wait()