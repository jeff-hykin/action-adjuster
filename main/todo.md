- figure out how the Experimental approach is outperforming the optimal when noise is added
    - check the velocity/spin units (they get scaled)
    - check if the prev_relative_velocity is effected by performing more granular updates
    - figure out why the prefect answer has a non-zero loss
        - map the design
- check if 0.1 loss with noise
- try using 0.001 for action duration

- figure how much noise it can handle and where
- zero-out the intial point
- optimization could help performance, options are:
    - add gradient decent tracking
    - vectorize/optimize the objective function
    - find/make a better implementation of cmaes
- test how much it improves policy reward
    - perform tons of tests (30 runs each)
        - DONE: compare NO_ADJUST, ADJUST, ORACLE with 10000 history size, heavy noise
            - variance (individual plots)
            - average
            - median
        - DONE: compare NO_ADJUST, ADJUST, ORACLE with 50 history size, heavy noise
            - variance (individual plots)
            - average
            - median
        - compare with 10000 history size, no noise
        - non linear adverse
        - one layer network
        - theorical
        
    - case where the perfect transformation shifts over time
    - case with no noise
    
- evaluate the difficulty of comparing to the meta learning paper
- look up Trim control (for aircraft)
- add lookbehind limiter, or maybe log
- record noise performance
- add sliding test

- finish removing magic numbers
- add new logic for "max_velocity_reset_number"
- figure out how to normalize the features in the spacial information vector
- DONE: add a summary evaluation metric (reward total)

Experimental
- combine update frequency and step size into one hyperparameter for action adjuster
- try using all the predicted points instead of just the far-out predicted values


- DONE: implement the real policy instead of going straight, tensorflow v1 is too hard to
- DONE: create a switch for publishing rospy stuff
- DONE: ros:
    - find and process observation data
    - publish control commands
- DONE: test ROS imports on linux machine
- DONE create the warthog_faker for testing the ROS code
- DONE: try to make fake ROS server
- DONE: start at the first waypoint, not a random waypoint