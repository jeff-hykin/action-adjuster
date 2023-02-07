import torch
from blissful_basics import to_pure, countdown, print
import numpy

from config import config, path_to
from envs.warthog import WarthogEnv
from tools.geometry import get_distance, get_angle_from_origin, zero_to_2pi, pi_to_pi, abs_angle_difference

# 
# models
# 
generate_next_spacial_info = WarthogEnv.sim_warthog
generate_next_observation  = WarthogEnv.generate_observation

# FIXME: replace this with a normalization method
spacial_coefficients = WarthogEnv.SpacialInformation([
    1, # x
    1, # y
    1, # angle
    0.3, # velocity # arbitraryly picked value 
    0.3, # spin # arbitraryly picked value 
])
class ActionAdjuster:
    def __init__(self, policy, initial_transform=None):
        self.policy = policy
        self.actual_spatial_values = []
        self.input_data            = []
        self.transform             = initial_transform
        self.should_update = countdown(config.action_adjuster.update_frequency)
        self.optimizer = None # will exist after first data is added
    
    def adjust(self, action, transform=None):
        if type(self.transform) == type(None):
            self.transform = numpy.ones((len(action),len(action)))
        if type(transform) == type(None):
            transform = self.transform
        
        return to_pure(numpy.array(action) * transform)
    
    def add_data(self, observation, additional_info):
        self.input_data.append(dict(
            policy=self.policy,
            observation=observation,
            additional_info=[
                additional_info["spacial_info_with_noise"],
                additional_info["remaining_waypoints"],
                additional_info["horizon"],
                additional_info["action_duration"],
            ],
        ))
        self.actual_spatial_values.append(additional_info["spacial_info_with_noise"])
        # NOTE: this^ is only grabbing the last, it would be valid to grab all of them
        # future work here might try grabbing all of them, or reducing the length, especially if the model/policy is thought to be bad/noisy
        if self.optimizer == None:
            action = self.policy(observation)
            # create transform if needed
            if type(self.transform) == type(None):
                self.transform = numpy.ones((len(action),len(action)))
        
        if self.should_update():
            self.fit_points()
    
    def fit_points(self):
        # no data
        if len(self.input_data) == 0:
            return
        
        self.optimizer
        relevent_observations = self.actual_spatial_values[config.action_adjuster.future_projection_length:]
        def objective_function(transform):
            predicted_spatial_values = []
            for each_input_data in self.input_data:
                spacial_expectation, observation_expectation = self.project(
                    transform=transform,
                    **each_input_data,
                )
                predicted_spatial_values.append(spacial_expectation[-1]) 
            
            # loss function
            loss = 0
            for each_actual, each_predicted in zip(relevent_observations, predicted_spatial_values):
                x1, y1, angle1, velocity1, spin1 = each_actual
                x2, y2, angle2, velocity2, spin2 = each_predicted
                iteration_total = 0
                iteration_total += spacial_coefficients.x        * abs((x1        - x2       ))
                iteration_total += spacial_coefficients.y        * abs((y1        - y2       ))
                iteration_total += spacial_coefficients.angle    * (abs_angle_difference(angle1, angle2)) # angle is different cause it wraps (0 == 2π)
                iteration_total += spacial_coefficients.velocity * abs((velocity1 - velocity2))
                iteration_total += spacial_coefficients.spin     * abs((spin1     - spin2    ))
                
                loss += iteration_total
            return loss
        
        print("action adjuster is updating transform matrix")
        self.transform = self.transform + (config.action_adjuster.update_rate * guess_to_maximize(objective_function, initial_guess=self.transform))
    
    # returns twos list, one of projected spacial_info's one of projected observations
    def project(self, policy, observation, additional_info, transform=None):
        current_spatial_info, remaining_waypoints, horizon, action_duration = additional_info
        observation_expectation = []
        spacial_expectation = []
        for each in range(config.action_adjuster.future_projection_length):
            action = policy(observation)
            velocity_action, spin_action = self.adjust(action, transform)
            
            next_spacial_info = generate_next_spacial_info(
                old_spatial_info=current_spatial_info,
                velocity_action=velocity_action,
                spin_action=spin_action,
                action_duration=action_duration,
            )
            next_observation = generate_next_observation(
                remaining_waypoints=remaining_waypoints,
                horizon=horizon,
                current_spacial_info=next_spacial_info,
            )
            spacial_expectation.append(next_spacial_info)
            observation_expectation.append(next_observation)
            
            observation          = next_observation
            current_spacial_info = next_spacial_info
        
        return spacial_expectation, observation_expectation

def guess_to_maximize(objective_function, initial_guess, stdev=1):
    import cma
    if len(initial_guess) == 1:
        new_objective = lambda arg1: objective_function(arg1[0])
    
    import sys
    
    original_stdout = sys.stdout
    with open("/dev/null", "w+") as null:
        sys.stdout = null
        try:
            xopt, es = cma.fmin2(
                lambda *args: -objective_function(*args),
                initial_guess,
                stdev,
                options=dict(
                    verb_log=0                      ,
                    verbose=0                       ,
                    verb_plot=0                     ,
                    verb_disp=0                     ,
                    verb_filenameprefix="/dev/null" ,
                    verb_append=0                   ,
                    verb_time=False                 ,
                ),
            )
        except Exception as error:
            raise error
        finally:
            sys.stdout = original_stdout
        
    if len(initial_guess) == 1:
        return xopt[0]
        
    return xopt