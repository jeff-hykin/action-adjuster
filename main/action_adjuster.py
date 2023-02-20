import torch
import trivial_torch_tools as ttt
from trivial_torch_tools import to_tensor
from blissful_basics import to_pure, countdown, print
import blissful_basics as bb
import numpy
from icecream import ic
ic.configureOutput(includeContext=True)

from config import config, path_to
from envs.warthog import WarthogEnv
from tools.geometry import get_distance, get_angle_from_origin, zero_to_2pi, pi_to_pi, abs_angle_difference

perfect_answer = to_tensor([
    [ 1,     0,    config.simulator.velocity_offset, ],
    [ 0,     1,        config.simulator.spin_offset, ],
    [ 0,     0,                                   1, ],
])

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
        self._transform            = None
        self.inverse_transform     = None
        self.transform             = initial_transform
        self.should_update = countdown(config.action_adjuster.update_frequency)
    
    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, value):
        if isinstance(value, type(None)):
            self._transform = None
        elif isinstance(value, torch.Tensor):
            self._transform = value
        else:
            self._transform = to_tensor(value)
        if type(self._transform) != type(None):
            self.inverse_transform = torch.linalg.inv(self._transform)
    
    def adjust(self, action, transform=None, real_transformation=True):
        if config.action_adjuster.disabled:
            return action # no change
        
        self._init_transform_if_needed(len(action))
        if type(transform) == type(None):
            transform = self.transform
        
        # the real transformation needs to compensate (inverse of curve-fitter transform)
        if real_transformation:
            *result, constant = torch.inner(
                to_tensor([*action, 1]),
                self.inverse_transform, 
            )
            return to_tensor(result)
        
        # the curve-fitter is finding the transform that would make the observed-trajectory match the model-predicted trajectory
        # (e.g. what transform is the world doing to our actions; once we know that we can compensate with and equal-and-opposite transformation)
        else:
            *result, constant = torch.inner(
                to_tensor([*action, 1]),
                self.transform,
            )
            return to_tensor(result)
    
    def _init_transform_if_needed(self, action_length):
        if type(self.transform) == type(None):
            self.transform = torch.eye(action_length+1)
    
    def add_data(self, observation, additional_info):
        if config.action_adjuster.disabled:
            return # no change
            
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
        
        action = self.policy(observation)
        # create transform if needed
        self._init_transform_if_needed(len(action))
        
        if self.should_update():
            self.fit_points()
    
    @property
    def readable_transform(self):
        return "[ " + ", ".join([ f"{each}" for each in to_pure(self.transform)]) + " ]"
    
    def fit_points(self):
        # no data
        if len(self.input_data) == 0:
            return
        
        lookback_size = (2*-config.action_adjuster.update_frequency)
        recent_data = self.input_data[lookback_size:]
        relevent_observations = self.actual_spatial_values[config.action_adjuster.future_projection_length:]
        def objective_function(transform):
            predicted_spatial_values = []
            for each_input_data in self.input_data:
                spacial_expectation, observation_expectation = self.project(
                    transform=transform,
                    real_transformation=False,
                    **each_input_data,
                )
                predicted_spatial_values.append(spacial_expectation[-1]) 
            
            # loss function
            loss = to_tensor(0.0).requires_grad_()
            for each_actual, each_predicted in zip(relevent_observations, predicted_spatial_values):
                x1, y1, angle1, velocity1, spin1 = each_actual
                x2, y2, angle2, velocity2, spin2 = each_predicted
                iteration_total = to_tensor(0.0).requires_grad_()
                iteration_total += spacial_coefficients.x        * torch.abs((x1        - x2       ))
                iteration_total += spacial_coefficients.y        * torch.abs((y1        - y2       ))
                iteration_total += spacial_coefficients.angle    * (abs_angle_difference(angle1, angle2)) # angle is different cause it wraps (0 == 2π)
                iteration_total += spacial_coefficients.velocity * torch.abs((velocity1 - velocity2))
                iteration_total += spacial_coefficients.spin     * torch.abs((spin1     - spin2    ))
                
                loss += iteration_total
            return -loss
        
        tracked_transform = to_tensor(self.transform).requires_grad_()
        score_before     = objective_function(tracked_transform)
        print(f'''score_before = {score_before}''')
        print(f'''score_before.backward() = {score_before.backward()}''')
        print(f'''tracked_transform.grad = {tracked_transform.grad}''')
        exit()
        
        transform_before = to_tensor(self.transform)
        score_before     = objective_function(self.transform)
        print("")
        print(f"action adjuster is updating transform vector (before): {self.readable_transform}, score:{score_before:.3f}")
        self.transform = self.transform + (config.action_adjuster.update_rate * guess_to_maximize(objective_function, initial_guess=self.transform, stdev=0.01))
        score_after = objective_function(self.transform)
        if score_after < score_before:
            # not only DONT use the new transform, but actually decay/cool-down the previous one
            # FIXME: this shouldnt shrink everything, it should instead get closer to the original (idenity) matrix
            self.transform = transform_before * config.action_adjuster.decay_rate
        print(f"action adjuster is updating transform vector (after ): {self.readable_transform}, score:{objective_function(self.transform):.3f}")
    
    # returns twos list, one of projected spacial_info's one of projected observations
    def project(self, policy, observation, additional_info, transform=None, real_transformation=True):
        current_spatial_info, remaining_waypoints, horizon, action_duration = additional_info
        observation_expectation = []
        spacial_expectation = []
        for each in range(config.action_adjuster.future_projection_length):
            action = policy(observation)
            velocity_action, spin_action = self.adjust(action, transform, real_transformation=real_transformation)
            
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
        
        return to_tensor(spacial_expectation), to_tensor(observation_expectation)

def guess_to_maximize(objective_function, initial_guess, stdev=1):
    import cma
    is_scalar = not bb.is_iterable(initial_guess)
    new_objective = objective_function
    if is_scalar: # wrap it
        new_objective = lambda arg1: objective_function(arg1[0])
        initial_guess = [initial_guess, 0]
    else: # flatten it
        initial_guess = to_tensor(initial_guess)
        shape = initial_guess.shape
        if shape == (1,):
            new_objective = lambda arg1: objective_function(to_tensor([arg1[0]]))
        elif len(shape) > 1:
            new_objective = lambda arg1: objective_function(arg1.reshape(shape))
    
    import sys
    
    original_stdout = sys.stdout
    with open("/dev/null", "w+") as null:
        sys.stdout = null
        try:
            xopt, es = cma.fmin2(
                lambda *args: -new_objective(*args),
                to_tensor(initial_guess.flat),
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
    
    output = xopt
    if is_scalar: # wrap it
        return output[0]
    else: # un-flatten it
        if shape == (1,):
            return to_tensor([output[0]])
        elif len(shape) > 1:
            return output.reshape(shape)
    
    return output