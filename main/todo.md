- zero-out the intial point
- test how much it improves policy reward
- use websockets instead of file-passing for async update

    
- evaluate the difficulty of comparing to the meta learning paper
- look up Trim control (for aircraft)
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
- DONE: ros:
    - find and process observation data
    - publish control commands
- DONE: test ROS imports on linux machine
- DONE create the warthog_faker for testing the ROS code
- DONE: try to make fake ROS server
- DONE: start at the first waypoint, not a random waypoint