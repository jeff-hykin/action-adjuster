import subprocess
import json
for each_profile in [ "@NO_ADJUSTER", "@PERFECT_ADJUSTER", "@NORMAL_ADJUSTER" ]:
    for run_number in range(30):
        output_folder = f"./records/{each_profile}|{run_number}"
        print(f"working on: {output_folder}")
        subprocess.check_output([ sys.executable, "main.py", each_profile, f"output_folder:{json.dumps(output_folder)}",  ]).decode('utf-8')[0:-1]