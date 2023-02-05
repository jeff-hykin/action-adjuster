magic_number_4  = 4           # FIXME: import these from somewhere once you know what they are
magic_number_2_point_5  = 2.5 # FIXME: import these from somewhere once you know what they are

def predict_next_spacial_info(current_spatial_info, velocity_action, spin_action, action_duration):
    import numpy as np
    import math
    from envs.warthog import WarthogEnv
    
    velocity_action = np.clip(velocity_action,  0, 1) * magic_number_4 # TODO: is it supposed to be clipped to self.max_velocity?
    spin_action     = np.clip(spin_action,     -1, 1) * magic_number_2_point_5
    
    old_velocity = current_spatial_info.velocity
    old_spin     = current_spatial_info.spin
    old_x        = current_spatial_info.x
    old_y        = current_spatial_info.y
    old_angle    = current_spatial_info.angle
    
    new_spacial_info = WarthogEnv.SpacialInformation(current_spatial_info)
    
    new_spacial_info.velocity = velocity_action
    new_spacial_info.spin     = spin_action
    
    print("old_x", old_x)
    print("old_velocity", old_velocity)
    print("math.cos(old_angle)", math.cos(old_angle))
    print("action_duration", action_duration)
    new_spacial_info.x          = old_x + old_velocity * math.cos(old_angle) * action_duration
    new_spacial_info.y          = old_y + old_velocity * math.sin(old_angle) * action_duration
    new_spacial_info.angle      = zero_to_2pi(old_angle + old_spin * action_duration)
    
    return new_spacial_info

def predict_next_observation_and_spacial_info(current_spatial_info, velocity_action, spin_action, additional_info):
    remaining_waypoints, horizon, action_duration = additional_info
    
    next_spacial_info = predict_next_spacial_info(current_spatial_info, velocity_action, spin_action, action_duration)
    return next_spacial_info, generate_observation(remaining_waypoints, horizon, next_spacial_info)

# returns a list of [horizon] observations (e.g. 10x10 waypoints)
def project(policy, observation, additional_info):
    current_spatial_info, remaining_waypoints, horizon, action_duration = additional_info
    observation_expectation = []
    for each in range(horizon):
        velocity_action, spin_action = policy(observation)
        current_spatial_info, observation = predict_next_observation_and_spacial_info(
            current_spatial_info,
            velocity_action,
            spin_action,
            [remaining_waypoints, horizon, action_duration],
        )
        observation_expectation.append(observation)
    
    return observation_expectation

def prediction_loss(actual_observations, projected_observations):
    loss = 0
    for actual, projected in zip(actual_observations, projected_observations):
        # FIXME: defined mean_squared_error
        loss += mean_squared_error(actual, projected)
    
    return loss

import torch
from blissful_basics import to_pure
class ActionAdjuster:
    def __init__(self, policy, initial_transform):
        self.policy = policy
        self.transform = torch.tensor(initial_transform)
        self.observations = []
        self.projections = []
    
    def add_data(self, observation, additional_info):
        self.observations.append(observation)
        # FIXME: add a trigger here to run the sovler
        
        # TODO: figure out the best way to create data for a linear/logrithmic/quadratic sovler
        self.projections.append(
            project(
                policy=self.policy,
                observation=observation,
                additional_info=[
                    additional_info["spacial_info_with_noise"],
                    additional_info["remaining_waypoints"],
                    additional_info["horizon"],
                    additional_info["action_duration"],
                ],
            )
        )
    
    def adjust(self, action):
        return to_pure(torch.tensor(action) * self.transform)


# 
# helpers
# 
def generate_observation(remaining_waypoints, horizon, current_spatial_info):
    magic_number_4 = 4 # I think this is len([x,y,spin,velocity])
    magic_number_2 = 2 # TODO: I have no idea what this 2 is for
    obs = [0] * ((horizon * magic_number_4) + magic_number_2)
    original_velocity = current_spatial_info.velocity
    original_spin     = current_spatial_info.spin
    
    closest_index = get_closest_index(remaining_waypoints, current_spatial_info.x, current_spatial_info.y)
    
    observation_index = 0
    for horizon_index in range(0, horizon):
        waypoint_index = horizon_index + closest_index
        if waypoint_index < len(remaining_waypoints):
            waypoint = remaining_waypoints[waypoint_index]
            x_diff = waypoint.x - current_spatial_info.x
            y_diff = waypoint.y - current_spatial_info.y
            radius = get_distance(waypoint.x, waypoint.y, current_spatial_info.x, current_spatial_info.y)
            angle_to_next_point = get_angle_from_origin(x_diff, y_diff)
            current_angle       = zero_to_2pi(current_spatial_info.angle)
            
            yaw_error = pi_to_pi(waypoint.angle - current_angle)
            velocity = waypoint.velocity
            obs[observation_index + 0] = radius
            obs[observation_index + 1] = pi_to_pi(angle_to_next_point - current_angle)
            obs[observation_index + 2] = yaw_error
            obs[observation_index + 3] = velocity - original_velocity
        else:
            obs[observation_index + 0] = 0.0
            obs[observation_index + 1] = 0.0
            obs[observation_index + 2] = 0.0
            obs[observation_index + 3] = 0.0
        observation_index = observation_index + magic_number_4
    obs[observation_index] = original_velocity
    obs[observation_index + 1] = original_spin
    return obs

def get_closest_index(remaining_waypoints, x, y):
    import math
    closest_index = 0
    closest_distance = math.inf
    for index in range(0, len(remaining_waypoints)):
        waypoint = remaining_waypoints[index]
        distance = get_distance(waypoint.x, waypoint.y, x, y)
        if distance <= closest_distance:
            closest_distance = distance
            closest_index = index
        else:
            break
    return closest_index


def get_distance(x1, y1, x2, y2):
    import math
    x_diff = x2 - x1
    y_diff = y2 - y1
    return math.sqrt(x_diff * x_diff + y_diff * y_diff)

def get_angle_from_origin(x, y):
    import math
    theta = math.atan2(y, x)
    return zero_to_2pi(theta)

def zero_to_2pi(theta):
    import math
    if theta < 0:
        theta = 2 * math.pi + theta
    elif theta > 2 * math.pi:
        theta = theta - 2 * math.pi
    return theta

def pi_to_pi(theta):
    import math
    if theta < -math.pi:
        theta = theta + 2 * math.pi
    elif theta > math.pi:
        theta = theta - 2 * math.pi
    return theta