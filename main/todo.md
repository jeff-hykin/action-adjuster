- confirm that self.input_data corrisponds with predicted spatial data
- make sure that the action that is saved (the record used in the objective function) accounts for the adjustment
- try sqrt instead of squaring loss
- try 1-step prediction instead of 10-step prediction

- finish removing magic numbers
- add new logic for "max_velocity_reset_number"
- figure out how to normalize the features in the spacial information vector
- add a summary evaluation metric (reward total)
- pull in a real policy

Experimental
- combine update frequency and step size into one hyperparameter for action adjuster
- try using all the predicted points instead of just the far-out predicted values