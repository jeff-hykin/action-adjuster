import sys
import subprocess
import json
from config import path_to
from subprocess import Popen, PIPE


experiment_number = 18
if __name__ == "__main__":
    finished = 0
    number_of_episode_runs_per_scenario = 15
    for run_number in range(finished, number_of_episode_runs_per_scenario):
        processes = []
        for each_profile in [ "@NORMAL_ADJUSTER", "@NO_ADJUSTER", "@PERFECT_ADJUSTER", ]:
            run_number_string = f"{run_number}".rjust(3,' ')
            output_folder = f"{path_to.records}/{experiment_number}.{each_profile}|{run_number_string}.ignore"
            profiles = [ each_profile, "@WARTHOG", "@NOISE=MEDIUM", "@ADVERSITY=STRONG", "@BATTERY_DRAIN" ]
            print(f"working on: {output_folder}")
            process = Popen([ sys.executable, path_to.main, *profiles, f"output_folder:{json.dumps(output_folder)}" ])
            processes.append(process)
        
        for each_process in processes:
            each_process.wait()
        # for some reason the processes don't seem to stop even after being waited on
        for each_process in processes:
            try: each_process.kill()
            except: pass
        for each_process in processes:
            try: each_process.terminate()
            except: pass