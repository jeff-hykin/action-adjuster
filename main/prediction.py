from config import config, path_to
from envs.warthog import WarthogEnv
from blissful_basics import countdown

# 
# models
# 
generate_next_spacial_info = WarthogEnv.sim_warthog
generate_next_observation  = WarthogEnv.generate_observation

import torch
from blissful_basics import to_pure
class ActionAdjuster:
    def __init__(self, policy, initial_transform):
        self.policy = policy
        self.transform = torch.tensor(initial_transform)
        self.actual_spatial_values = []
        self.input_data            = []
        self.should_update = countdown(config.update_frequency)
    
    def adjust(self, action, transform=None):
        if type(transform) == type(None):
            transform = self.transform
        
        return to_pure(torch.tensor(action) * transform)
    
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
        
        if self.should_update():
            self.fit_points()
    
    def fit_points(self):
        relevent_observations = self.actual_spatial_values[config.action_adjuster.future_projection_length:]
        def objective_function(transform):
            predicted_spatial_values = []
            for each_input_data in self.input_data:
                spacial_expectation, observation_expectation = project(
                    transform=transform,
                    **each_input_data,
                )
                predicted_spatial_values.append(spacial_expectation[-1]) 
            
            # loss function
            loss = 0
            for each_actual, each_predicted in zip(relevent_observations, predicted_spatial_values):
                # FIXME: actual loss function
                pass
            return loss
            
                
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