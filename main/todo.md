- ros:
    - find and process observation data
    - publish control commands
    - create a listener
    - finish rospy:
        simulate listener: /Users/jeffhykin/repos/action-adjuster/.ignore.repo/warthog_rl/ranger_sim_data_collecter_steering.py
    
- evaluate the difficulty of comparing to the meta learning paper
- look up Trim control (for aircraft)
- create a file system passthrough to asyncly update things
    - have action adjuster add-info log observations to a folder
    - have seperate task read from folder, run the solver, then solved, write to `best_canidate.json`
- add lookbehind limiter, or maybe log
- record noise performance
- add sliding test
- try sqrt instead of squaring loss

- finish removing magic numbers
- add new logic for "max_velocity_reset_number"
- figure out how to normalize the features in the spacial information vector
- DONE: add a summary evaluation metric (reward total)

Experimental
- combine update frequency and step size into one hyperparameter for action adjuster
- try using all the predicted points instead of just the far-out predicted values


- DONE: implement the real policy instead of going straight, tensorflow v1 is too hard to
- DONE: create a switch for publishing rospy stuff