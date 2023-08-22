import sys
import subprocess
import json
from config import path_to, config, send_notification
from subprocess import Popen, PIPE
from __dependencies__.blissful_basics import FS, run_main_hooks_if_needed
from __dependencies__.informative_iterator import ProgressBar

if __name__ == "__main__":
    from plots.main_comparision import main as generate_plots
    run_main_hooks_if_needed(__name__)
    
    finished = 0
    for progress, run_number in ProgressBar(tuple(range(finished, config.number_of_episode_runs_per_scenario))):
        if progress.index != 0: send_notification(progress.previous_output)
        processes = []
        for each_profile in [ "@NORMAL_ADJUSTER", "@NO_ADJUSTER", "@PERFECT_ADJUSTER", ]:
            run_number_string = f"{run_number}".rjust(3,' ')
            output_folder = f"{path_to.records}/{config.experiment_number}.{each_profile}|{run_number_string}.ignore"
            profiles = [ each_profile, "@WARTHOG", "@NOISE=MEDIUM", "@ADVERSITY=STRONG", "@BATTERY_DRAIN" ]
            print(f"working on: {output_folder}")
            process = Popen([ sys.executable, path_to.main, *profiles, f"output_folder:{json.dumps(output_folder)}" ])
            processes.append(process)
        
        for each_process in processes:
            each_process.wait()
            
        try: generate_plots(display=False)
        except Exception as error:
            pass
        
    # start generating the graphs so they're cached
    send_notification(progress.previous_output)