import math
import time
import random
from collections import namedtuple
import json

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import numpy
import numpy as np
import gym

import __dependencies__.blissful_basics as bb
from __dependencies__.super_hash import super_hash
from __dependencies__.blissful_basics import Csv, create_named_list_class, FS, print, stringify, clip, countdown, LazyDict
from __dependencies__.grug_test import yaml, ez_yaml, register_named_tuple

from config import config, path_to, grug_test
from generic_tools.geometry import get_distance, get_angle_from_origin, zero_to_2pi, pi_to_pi, abs_angle_difference, angle_created_by
from data_structures import Unknown, Action, StepOutput, StepSideEffects, GetObservationOutput, RewardOutput, SimWarthogOutput, PoseEntry, TwistEntry, SpacialHistory, SpacialInformation, ReactionClass, WaypointGap, Waypoint, Observation, AdditionalInfo
from render import Renderer
from misc import scaled_sigmoid

# bb.Warnings.disable()
magic_number_1_point_5 = 1.5
magic_number_1_point_4 = 1.4

@grug_test(max_io=10, skip=True)
def read_waypoint_file(filename):
    comments, column_names, rows = Csv.read(filename, separator=",", first_row_is_column_names=True, skip_empty_lines=True)
    desired_velocities = []
    waypoints_list = []
    min_x = min(row.x for row in rows)
    min_y = min(row.y for row in rows)
    for row in rows:
        desired_velocities.append(row.velocity)
        waypoints_list.append(
            Waypoint([row.x-min_x, row.y-min_y, row.angle, row.velocity])
        )
    
    index = 0
    for index in range(0, len(waypoints_list) - 1):
        current_waypoint = waypoints_list[index]
        next_waypoint    = waypoints_list[index+1]
        
        x_diff = next_waypoint.x - current_waypoint.x # meters
        y_diff = next_waypoint.y - current_waypoint.y # meters
        # angle radians
        current_waypoint.angle = zero_to_2pi(get_angle_from_origin(x_diff, y_diff))
    
    assert waypoints_list[index+1].tolist() == waypoints_list[-1].tolist()
    final_waypoint = waypoints_list[index + 1]
    final_waypoint.angle = waypoints_list[-2].angle # fill in the blank value
    
    return desired_velocities, waypoints_list


class WarthogEnv(gym.Env):
    random_start_position_offset = config.simulator.random_start_position_offset
    random_start_angle_offset    = config.simulator.random_start_angle_offset
    max_relative_velocity = 1
    min_relative_velocity = 0
    max_relative_spin = 1
    min_relative_spin = -1
    
    action_space = gym.spaces.Box(
        low=np.array(config.simulator.action_space.low),
        high=np.array(config.simulator.action_space.high),
        shape=np.array(config.simulator.action_space.low).shape,
    )
    observation_space = gym.spaces.Box(
        low=config.simulator.observation_space.low,
        high=config.simulator.observation_space.high,
        shape=config.simulator.observation_space.shape,
        dtype=float,
    )
    
    def __init__(self, waypoint_file_path, trajectory_output_path=None, recorder=None):
        super(WarthogEnv, self).__init__()
        self._ = LazyDict()
        self.log = LazyDict()
        if True:
            self.waypoint_file_path = waypoint_file_path
            self.trajectory_output_path = trajectory_output_path
            self.recorder = recorder
            
            self.waypoints_list   = []
            self.prev_spacial_info = SpacialInformation(0,0,0,0,0,-1)
            self.prev_spacial_info_with_noise = SpacialInformation(0,0,0,0,0,-1)
            self.spacial_info_with_noise = SpacialInformation(0,0,0,0,0,0)
            self.spacial_info = SpacialInformation(
                x                = 0,
                y                = 0,
                angle            = 0,
                velocity         = 0,
                spin             = 0,
                timestep         = 0,
            )
            self.observation = None
            
            self.max_number_of_timesteps_per_episode      = config.simulator.max_number_of_timesteps_per_episode
            self.save_data              = config.simulator.save_data
            self.action_duration        = config.simulator.action_duration  
            self.next_waypoint_index    = 0
            self.prev_next_waypoint_index     = 0
            self.closest_distance       = math.inf
            self.desired_velocities     = []
            self.episode_steps          = 0
            self.total_episode_reward   = 0
            self.reward                 = 0
            self.original_relative_spin           = 0 
            self.original_relative_velocity       = 0 
            self.prev_original_relative_spin      = 0 
            self.prev_original_relative_velocity  = 0 # "original" is what the actor said to do
            self.mutated_relative_spin            = 0 # "mutated" is after adversity+noise was added
            self.mutated_relative_velocity        = 0
            self.prev_mutated_relative_spin       = 0
            self.prev_mutated_relative_velocity   = 0
            self.is_episode_start        = 1
            self.trajectory_file         = None
            self.global_timestep         = 0
            self.action_buffer           = [ (0,0) ] * config.simulator.action_delay # seed the buffer with delays
            self.simulated_battery_level = 1.0 # proportion 
            
            if self.waypoint_file_path is not None:
                self.desired_velocities, self.waypoints_list = read_waypoint_file(self.waypoint_file_path)
            
            self.crosstrack_error = 0
            self.velocity_error   = 0
            self.phi_error        = 0
            
            # 
            # trajectory_file
            # 
            if self.trajectory_output_path is not None:
                print(f'''trajectory being logged to: {trajectory_output_path}''')
                FS.ensure_is_folder(FS.parent_path(trajectory_output_path))
                self.trajectory_file = open(trajectory_output_path, "w+")
                self.trajectory_file.writelines(f"x, y, angle, velocity, spin, velocity_action, spin_action, is_episode_start\n")
            
            self.reset()
            self._ = LazyDict({
                "waypoints_list"                     : getattr(self, "waypoints_list", None),
                "pose"                               : getattr(self, "pose", None),
                "twist"                              : getattr(self, "twist", None),
                "next_waypoint_index_"               : getattr(self, "next_waypoint_index_", None),
                "prev_next_waypoint_index_"          : getattr(self, "prev_next_waypoint_index_", None),
                "closest_distance"                   : getattr(self, "closest_distance", None),
                "number_of_waypoints"                : getattr(self, "number_of_waypoints", None),
                "horizon"                            : getattr(self, "horizon", config.simulator.horizon),
                "action_duration"                    : getattr(self, "action_duration", None),
                "desired_velocities"                 : getattr(self, "desired_velocities", None),
                "max_vel"                            : getattr(self, "max_vel", None),
                "waypoints_dist"                     : getattr(self, "waypoints_dist", None),
                "warthog_length"                     : getattr(self, "warthog_length", None),
                "warthog_width"                      : getattr(self, "warthog_width", None),
                "diag_angle"                         : getattr(self, "diag_angle", None),
                "prev_angle"                         : getattr(self, "prev_angle", None),
                "crosstrack_error"                   : getattr(self, "crosstrack_error", None),
                "vel_error"                          : getattr(self, "vel_error", None),
                "phi_error"                          : getattr(self, "phi_error", None),
                "start_step_for_sup_data"            : getattr(self, "start_step_for_sup_data", None),
                "max_number_of_timesteps_per_episode": getattr(self, "max_number_of_timesteps_per_episode", None),
                "total_ep_reward"                    : getattr(self, "total_ep_reward", None),
                "reward"                             : getattr(self, "reward", None),
                "original_relative_spin"             : getattr(self, "original_relative_spin", None),
                "action"                             : getattr(self, "action", None),
                "relative_action"                    : getattr(self, "relative_action", None),
                "absolute_action"                    : getattr(self, "absolute_action", None),
                "prev_absolute_action"               : getattr(self, "prev_absolute_action", None),
                "omega_reward"                       : getattr(self, "omega_reward", None),
                "vel_reward"                         : getattr(self, "vel_reward", None),
                "delay_steps"                        : getattr(self, "delay_steps", None),
                "save_data"                          : getattr(self, "save_data", None),
                "ep_dist"                            : getattr(self, "ep_dist", None),
                "ep_poses"                           : getattr(self, "ep_poses", None),
            })
    
    def __del__(self):
        if self.trajectory_file:
            self.trajectory_file.close()
            
    @staticmethod
    @grug_test(max_io=30, record_io=None, additional_io_per_run=None, skip=True)
    def generate_observation(closest_index, remaining_waypoints, current_spacial_info):
        """
            Note:
                This function should be fully deterministic.
        """
        mutated_absolute_velocity = current_spacial_info.velocity
        mutated_absolute_spin     = current_spacial_info.spin
        
        # observation_length = (config.simulator.horizon*4)+3
        observation = []
        for waypoint in remaining_waypoints[0:config.simulator.horizon]:
            x_diff = waypoint.x - current_spacial_info.x
            y_diff = waypoint.y - current_spacial_info.y
            angle_to_next_point = get_angle_from_origin(x_diff, y_diff)
            current_angle       = zero_to_2pi(current_spacial_info.angle)
            
            gap_of_distance                    = get_distance(waypoint.x, waypoint.y, current_spacial_info.x, current_spacial_info.y)
            gap_of_angle_directly_towards_next = pi_to_pi(angle_to_next_point - current_angle)
            gap_of_desired_angle_at_next       = pi_to_pi(waypoint.angle      - current_angle)
            gap_of_velocity                    = waypoint.velocity - mutated_absolute_velocity
            
            observation.append(gap_of_distance)
            observation.append(gap_of_angle_directly_towards_next)
            observation.append(gap_of_desired_angle_at_next)
            observation.append(gap_of_velocity)
        
        while len(observation) < (config.simulator.horizon*4):
            observation.append(0)
        
        observation.append(mutated_absolute_velocity)
        observation.append(mutated_absolute_spin)
        observation.append(current_spacial_info.timestep)
        
        observation = Observation(observation)
        return observation
    
    @staticmethod
    @grug_test(max_io=30, record_io=None, additional_io_per_run=None, skip=False)
    def generate_next_spacial_info(old_spacial_info, relative_velocity, relative_spin, action_duration, debug=False, **kwargs):
        '''
            Note:
                This function should also be fully deterministic
            Inputs:
                relative_velocity: a value between 0 and 1, which will be scaled between 0 and controller_max_velocity
                relative_spin: a value between -1 and 1, which will be scaled between 0 and controller_max_velocity
        '''
        effective_action_duration = action_duration/config.simulator.granularity_of_calculations
        absolute_velocity = clip(relative_velocity, min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity) * config.vehicle.controller_max_velocity
        absolute_spin     = clip(relative_spin    , min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    ) * config.vehicle.controller_max_spin
        
        next_spacial_info = SpacialInformation(
            x=old_spacial_info.x,
            y=old_spacial_info.y,
            angle=old_spacial_info.angle,
            velocity=old_spacial_info.velocity,
            spin=old_spacial_info.spin,
            timestep=old_spacial_info.timestep+1,
        )
        
        
        # granularity substeps, having at least 3 of these steps is important
        for each in range(config.simulator.granularity_of_calculations):
            old_absolute_velocity = old_spacial_info.velocity
            old_absolute_spin     = old_spacial_info.spin
            old_x                 = old_spacial_info.x
            old_y                 = old_spacial_info.y
            old_angle             = old_spacial_info.angle
        
            if debug:
                with print.indent:
                    print("""next_spacial_info.x        = {old_x} + {old_absolute_velocity} * {math.cos(old_angle)} * {effective_action_duration}""")
                    print(f"""next_spacial_info.x        = {old_x} + {old_absolute_velocity} * {math.cos(old_angle)} * {effective_action_duration}""")
                    print(f"""next_spacial_info.x        = {old_x} + {old_absolute_velocity * math.cos(old_angle)} * {effective_action_duration}""")
                    print(f"""next_spacial_info.x        = {old_x} + {old_absolute_velocity * math.cos(old_angle) * effective_action_duration}""")
                    print()
                    print("""next_spacial_info.y        = {old_y} + {old_absolute_velocity} * {math.sin(old_angle)} * {effective_action_duration}""")
                    print(f"""next_spacial_info.y        = {old_y} + {old_absolute_velocity} * {math.sin(old_angle)} * {effective_action_duration}""")
                    print(f"""next_spacial_info.y        = {old_y} + {old_absolute_velocity * math.sin(old_angle)} * {effective_action_duration}""")
                    print(f"""next_spacial_info.y        = {old_y} + {old_absolute_velocity * math.sin(old_angle) * effective_action_duration}""")
                    print()
                    print("""next_spacial_info.angle    = zero_to_2pi(old_angle + old_absolute_spin           * effective_action_duration)""")
                    print(f"""next_spacial_info.angle    = zero_to_2pi({old_angle} + {old_absolute_spin}           * {effective_action_duration})""")
                    print(f"""next_spacial_info.angle    = zero_to_2pi({old_angle} + {old_absolute_spin * effective_action_duration})""")
                    print(f"""next_spacial_info.angle    = zero_to_2pi({old_angle + old_absolute_spin * effective_action_duration})""")
                    print(f"""next_spacial_info.angle    = {zero_to_2pi(old_angle + old_absolute_spin * effective_action_duration)}""")
                    print()
                    print(f'''next_spacial_info.velocity = {absolute_velocity}''')
                    print(f'''next_spacial_info.spin = {absolute_spin}''')
                
            next_spacial_info = SpacialInformation(
                velocity = absolute_velocity,
                spin     = absolute_spin,
                x        = old_x + old_absolute_velocity * math.cos(old_angle) * effective_action_duration,
                y        = old_y + old_absolute_velocity * math.sin(old_angle) * effective_action_duration,
                angle    = zero_to_2pi(old_angle + old_absolute_spin           * effective_action_duration),
                timestep = next_spacial_info.timestep,
            )    
            
            # repeat the process
            old_spacial_info = next_spacial_info
        
        if debug:
            with print.indent:
                print(f'''next_spacial_info = {next_spacial_info}''')
        return next_spacial_info
    
    @staticmethod
    @grug_test(max_io=30, record_io=None, additional_io_per_run=None, skip=True)
    def get_closest(remaining_waypoints, x, y):
        """
            Note:
                A helper for .generate_next_observation() and .reset()
        """
        closest_index = 0
        closest_distance = math.inf
        for index, waypoint in enumerate(remaining_waypoints):
            distance = get_distance(waypoint.x, waypoint.y, x, y)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = index
        return closest_index, closest_distance
    
    @staticmethod
    @grug_test(max_io=30, record_io=None, additional_io_per_run=None, skip=True)
    def original_reward_function(*, spacial_info, closest_distance, relative_velocity, prev_relative_velocity, relative_spin, prev_relative_spin, closest_waypoint, closest_relative_index,):
        x_diff     = closest_waypoint.x - spacial_info.x
        y_diff     = closest_waypoint.y - spacial_info.y
        angle_diff = get_angle_from_origin(x_diff, y_diff)
        yaw_error  = pi_to_pi(angle_diff - spacial_info.angle)

        velocity_error   = closest_waypoint.velocity - spacial_info.velocity
        crosstrack_error = closest_distance * math.sin(yaw_error)
        phi_error        = pi_to_pi(zero_to_2pi(closest_waypoint.angle) - spacial_info.angle)
        
        
        max_expected_crosstrack_error = config.reward_parameters.max_expected_crosstrack_error # meters
        max_expected_velocity_error   = config.reward_parameters.max_expected_velocity_error * config.vehicle.controller_max_velocity # meters per second
        max_expected_angle_error      = config.reward_parameters.max_expected_angle_error # 60° but in radians

        # base rewards
        crosstrack_reward = max_expected_crosstrack_error - math.fabs(crosstrack_error)
        velocity_reward   = max_expected_velocity_error   - math.fabs(velocity_error)
        angle_reward      = max_expected_angle_error      - math.fabs(phi_error)

        # combine
        running_reward = crosstrack_reward * velocity_reward * angle_reward

        # smoothing penalties (jerky=costly to machine and high energy/breaks consumption)
        running_reward -= config.reward_parameters.velocity_jerk_cost_coefficient * math.fabs(relative_velocity - prev_relative_velocity)
        running_reward -= config.reward_parameters.spin_jerk_cost_coefficient     * math.fabs(relative_spin - prev_relative_spin)
        # direct energy consumption
        running_reward -= config.reward_parameters.direct_velocity_cost * math.fabs(relative_velocity)
        running_reward -= config.reward_parameters.direct_spin_cost     * math.fabs(relative_spin)
        
        if config.reward_parameters.velocity_caps_enabled:
            abs_velocity_error = math.fabs(velocity_error)
            for min_waypoint_speed, max_error_allowed in config.reward_parameters.velocity_caps.items():
                # convert %'s to vehicle-specific values
                min_waypoint_speed = float(min_waypoint_speed.replace("%", ""))/100 * config.vehicle.controller_max_velocity
                max_error_allowed  = float(max_error_allowed.replace( "%", ""))/100 * config.vehicle.controller_max_velocity
                # if rule-is-active
                if closest_waypoint.velocity >= min_waypoint_speed: # old code: self.waypoints_list[k][3] >= 2.5
                    # if rule is broken, no reward
                    if abs_velocity_error > max_error_allowed: # old code: math.fabs(self.vel_error) > 1.5
                        running_reward = 0
                        break
        
        return running_reward, velocity_error, crosstrack_error, phi_error
    
    @staticmethod
    @grug_test(max_io=30, record_io=None, additional_io_per_run=None, skip=True)
    def almost_original_reward_function(**kwargs):
        closest_relative_index = kwargs["closest_relative_index"]
        running_reward, *other = WarthogEnv.original_reward_function(**kwargs)
        running_reward += closest_relative_index * config.reward_parameters.completed_waypoint_bonus
        return (running_reward, *other)
    
    @staticmethod
    def reward_function2(*, spacial_info, closest_distance, relative_velocity, prev_relative_velocity, relative_spin, prev_relative_spin, closest_waypoint, closest_relative_index):
        running_reward = 0
        
        # everything in this if block is the same as the original implementation
        if True:
            x_diff     = closest_waypoint.x - spacial_info.x
            y_diff     = closest_waypoint.y - spacial_info.y
            angle_diff = get_angle_from_origin(x_diff, y_diff)
            yaw_error  = pi_to_pi(angle_diff - spacial_info.angle)

            velocity_error   = closest_waypoint.velocity - spacial_info.velocity
            crosstrack_error = closest_distance * math.sin(yaw_error)
            phi_error        = pi_to_pi(closest_waypoint.angle - spacial_info.angle)
            
            
            max_expected_crosstrack_error = config.reward_parameters.max_expected_crosstrack_error # meters
            max_expected_velocity_error   = config.reward_parameters.max_expected_velocity_error * config.vehicle.controller_max_velocity # meters per second
            max_expected_angle_error      = config.reward_parameters.max_expected_angle_error # 60° but in radians

            # base rewards
            crosstrack_reward = max_expected_crosstrack_error - math.fabs(crosstrack_error)
            velocity_reward   = max_expected_velocity_error   - math.fabs(velocity_error)
            angle_reward      = max_expected_angle_error      - math.fabs(phi_error)
        
        advanced_reward = crosstrack_reward * velocity_reward * angle_reward
        
        # distance penalty
        distance   = math.sqrt(x_diff**2 + y_diff**2)
        one_to_zero_distance_penalty = (1 - scaled_sigmoid(distance))
        running_reward += config.reward_parameters.distance_scale * one_to_zero_distance_penalty

        # combine (only use advanced reward when close to point; e.g. proportionally scale advanced reward with closeness)
        running_reward += one_to_zero_distance_penalty * advanced_reward

        # smoothing penalties (jerky=costly to machine and high energy/breaks consumption)
        running_reward -= config.reward_parameters.velocity_jerk_cost_coefficient * math.fabs(relative_velocity - prev_relative_velocity)
        running_reward -= config.reward_parameters.spin_jerk_cost_coefficient     * math.fabs(relative_spin - prev_relative_spin)
        # direct energy consumption
        running_reward -= config.reward_parameters.direct_velocity_cost * math.fabs(relative_velocity)
        running_reward -= config.reward_parameters.direct_spin_cost     * math.fabs(relative_spin)
        
        # bonus for completing waypoints
        running_reward += closest_relative_index * config.reward_parameters.completed_waypoint_bonus
        
        return running_reward, velocity_error, crosstrack_error, phi_error
    
    def step(self, action, override_next_spacial_info=None):
        """
            Note:
                this is where all the noise (action noise, observation noise) is added.
                Note: noise is added before the values are scaled, which might amplify noise.
                
                `override_next_spacial_info` is currently only used with ROS runtime. This
                allows for most of the code to stay as-is while throwing away simulated data
                in favor of real-world data.
        """
        # 
        # push new action
        # 
        self.prev_original_relative_velocity = self.original_relative_velocity
        self.prev_original_relative_spin     = self.original_relative_spin
        self.prev_mutated_relative_velocity  = self.mutated_relative_velocity
        self.prev_mutated_relative_spin      = self.mutated_relative_spin
        self.original_relative_velocity, self.original_relative_spin = action
        self.absolute_action = [
            clip(self.original_relative_velocity, min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity) * config.vehicle.controller_max_velocity,
            clip(self.original_relative_spin    , min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    ) * config.vehicle.controller_max_spin
        ]
        
        # 
        # logging and counter-increments
        # 
        if self.save_data and self.trajectory_file is not None:
            self.trajectory_file.writelines(f"{self.spacial_info.x}, {self.spacial_info.y}, {self.spacial_info.angle}, {self.spacial_info.velocity}, {self.spacial_info.spin}, {self.original_relative_velocity}, {self.original_relative_spin}, {self.is_episode_start}\n")
        self.global_timestep += 1
        self.episode_steps = self.episode_steps + 1
        self.is_episode_start = 0
        
        # 
        # modify action
        # 
        if True:
            # first force them to be within normal ranges
            mutated_relative_velocity_action = clip(self.original_relative_velocity,  min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity)
            mutated_relative_spin_action     = clip(self.original_relative_spin    ,  min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    )
            
            # 
            # ADVERSITY
            # 
            if True:
                # battery adversity
                if config.simulator.battery_adversity_enabled:
                    self.simulated_battery_level *= 1-config.simulator.battery_decay_rate
                    self.recorder.add(timestep=self.global_timestep, simulated_battery_level=self.simulated_battery_level)
                    self.recorder.commit()
                    mutated_relative_velocity_action *= self.simulated_battery_level
                    # make sure velocity never goes negative (treat low battery as resistance)
                    mutated_relative_velocity_action = clip(mutated_relative_velocity_action,  min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity)
                
                # additive adversity
                mutated_relative_velocity_action += config.simulator.velocity_offset
                mutated_relative_spin_action     += config.simulator.spin_offset
        
            # 
            # add noise
            # 
            if config.simulator.use_gaussian_action_noise:
                mutated_relative_velocity_action += random.normalvariate(mu=0, sigma=config.simulator.gaussian_action_noise.velocity_action.standard_deviation, )
                mutated_relative_spin_action     += random.normalvariate(mu=0, sigma=config.simulator.gaussian_action_noise.spin_action.standard_deviation    , )
            
            # 
            # action delay
            # 
            self.action_buffer.append((mutated_relative_velocity_action, mutated_relative_spin_action))
            mutated_relative_velocity_action, mutated_relative_spin_action = self.action_buffer.pop(0) # ex: if 0 delay, this pop() will get what was just appended
            
            # 
            # save
            # 
            self.mutated_relative_velocity = mutated_relative_velocity_action
            self.mutated_relative_spin     = mutated_relative_spin_action
        
        
        # 
        # modify spacial_info
        # 
        self.prev_spacial_info = self.spacial_info
        if type(override_next_spacial_info) != type(None):
            # this is when the spacial_info is coming from the real world
            self.spacial_info = override_next_spacial_info
        else:
            # 
            # apply action
            # 
            self.spacial_info = WarthogEnv.generate_next_spacial_info(
                old_spacial_info=SpacialInformation(*self.spacial_info),
                relative_velocity=self.mutated_relative_velocity,
                relative_spin=self.mutated_relative_spin,
                action_duration=self.action_duration,
            )
        
        # 
        # increment waypoints
        # 
        if True:
            closest_relative_index = 0
            last_waypoint_index = len(self.waypoints_list)-1
            next_waypoint          = self.waypoints_list[self.next_waypoint_index]
            if len(self.waypoints_list) > 1:
                distance_to_waypoint       = get_distance(next_waypoint.x, next_waypoint.y, self.spacial_info.x, self.spacial_info.y)
                got_further_away           = self.closest_distance < distance_to_waypoint
                was_within_waypoint_radius = min(distance_to_waypoint, self.closest_distance) < config.simulator.waypoint_radius
                # went past waypoint? increment the index
                if distance_to_waypoint == 0 or (got_further_away and was_within_waypoint_radius):
                    closest_relative_index  = 1
                # went past waypoint, but edgecase of getting further away:
                elif was_within_waypoint_radius and self.next_waypoint_index < last_waypoint_index:
                    next_next_waypoint         = self.waypoints_list[self.next_waypoint_index+1]
                    waypoint_arm_angle = angle_created_by(
                        start=(self.spacial_info.x, self.spacial_info.y),
                        midpoint=(next_waypoint.x, next_waypoint.y),
                        end=(next_next_waypoint.x, next_next_waypoint.y),
                    )
                    we_passed_the_waypoint = waypoint_arm_angle < abs(math.degrees(90))
                    if we_passed_the_waypoint:
                        closest_relative_index  = 1
                        
            if closest_relative_index > 0:
                self.next_waypoint_index += 1
                # prevent indexing error
                self.next_waypoint_index = min(self.next_waypoint_index, len(self.waypoints_list)-1)
                next_waypoint = self.waypoints_list[self.next_waypoint_index]
            self.closest_distance = get_distance(
                next_waypoint.x,
                next_waypoint.y,
                self.spacial_info.x,
                self.spacial_info.y
            )
        
        # 
        # Reward Calculation
        # 
        self.reward, self.velocity_error, self.crosstrack_error, self.phi_error = WarthogEnv.almost_original_reward_function(
            spacial_info=self.spacial_info,
            closest_distance=self.closest_distance,
            relative_velocity=self.mutated_relative_velocity,
            prev_relative_velocity=self.prev_mutated_relative_velocity,
            relative_spin=self.mutated_relative_spin,
            prev_relative_spin=self.prev_mutated_relative_spin,
            closest_waypoint=next_waypoint,
            closest_relative_index=closest_relative_index,
        )
        
        # 
        # make mutated observation
        # 
        if True:
            self.prev_spacial_info_with_noise = self.spacial_info_with_noise
            self.spacial_info_with_noise = self.spacial_info
            # 
            # add spacial noise
            # 
            if config.simulator.use_gaussian_spacial_noise:
                self.spacial_info_with_noise = SpacialInformation(
                    self.spacial_info_with_noise.x        + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.x.standard_deviation       , ),
                    self.spacial_info_with_noise.y        + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.y.standard_deviation       , ),
                    self.spacial_info_with_noise.angle    + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.angle.standard_deviation   , ),
                    self.spacial_info_with_noise.velocity + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.spin.standard_deviation    , ),
                    self.spacial_info_with_noise.spin     + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.velocity.standard_deviation, ),
                    self.spacial_info_with_noise.timestep ,
                )
            
            # generate observation off potentially incorrect (noisey) spacial info
            prev_observation = self.observation
            self.observation = WarthogEnv.generate_observation(
                closest_index=self.next_waypoint_index,
                remaining_waypoints=self.waypoints_list[self.prev_next_waypoint_index:],
                current_spacial_info=self.spacial_info_with_noise,
            )
        
        # 
        # render
        # 
        self.renderer.render_if_needed(
            prev_next_waypoint_index=self.prev_next_waypoint_index,
            x_point=self.spacial_info.x, # self.spacial_info.x
            y_point=self.spacial_info.y, # self.spacial_info.y
            angle=self.spacial_info.angle,   # self.spacial_info.angle
            text_data=f"vel_error={self.velocity_error:.3f}\nclosest_index={self.next_waypoint_index}\ncrosstrack_error={self.crosstrack_error:.3f}\nReward={self.reward:.4f}\nwarthog_vel={self.spacial_info.velocity:.3f}\nphi_error={self.phi_error*180/math.pi:.4f}\nep_reward={self.total_episode_reward:.4f}\n\nomega_reward={(-2 * math.fabs(self.original_relative_spin)):.4f}\nvel_reward={self.velocity_error:.4f}",
        )
        
        additional_info = AdditionalInfo(
            timestep_index=Unknown,
            action_duration=self.action_duration,
            spacial_info=self.prev_spacial_info,
            spacial_info_with_noise=self.prev_spacial_info_with_noise,
            observation_from_spacial_info_with_noise=prev_observation,
            historic_transform=Unknown,
            original_reaction=ReactionClass(self.original_relative_velocity, self.original_relative_spin ),
            mutated_reaction=ReactionClass(self.mutated_relative_velocity, self.mutated_relative_spin ),
            next_spacial_info=self.spacial_info,
            next_spacial_info_spacial_info_with_noise=self.spacial_info_with_noise,
            next_observation_from_spacial_info_with_noise=self.observation,
            next_closest_index=self.next_waypoint_index,
            reward=self.reward,
        )
        
        # 
        # done Calculation
        #
        done = False
        if self.next_waypoint_index >= self.number_of_waypoints - 1:
            done = True 
        if self.episode_steps == self.max_number_of_timesteps_per_episode:
            done = True
            self.episode_steps = 0
        # immediate end due to too much loss
        if config.simulator.allow_cut_short_episode:
            if math.fabs(self.crosstrack_error) > magic_number_1_point_5 or math.fabs(self.phi_error) > magic_number_1_point_4:
                done = True
        
        return self.observation, self.reward, done, additional_info

    def reset(self, override_next_spacial_info=None):
        self.is_episode_start = 1
        self.total_episode_reward = 0
        
        index = config.simulator.starting_waypoint
        if config.simulator.starting_waypoint == 'random':
            assert self.number_of_waypoints > 21
            index = np.random.randint(self.number_of_waypoints-20, size=1)[0]
            
        # if position is overriden by (most likely) the real world position
        if type(override_next_spacial_info) != type(None):
            # this is when the spacial_info is coming from the real world
            self.spacial_info = override_next_spacial_info
            self.next_waypoint_index = index
            self.prev_next_waypoint_index = index
        # simulator position
        else:
            waypoint = self.waypoints_list[index]
            self.next_waypoint_index = index
            self.prev_next_waypoint_index = index
            
            self.spacial_info = SpacialInformation(
                x=waypoint.x + self.random_start_position_offset,
                y=waypoint.y + self.random_start_position_offset,
                angle=waypoint.angle + self.random_start_angle_offset,
                velocity=0,
                spin=0,
                timestep=0,
            )
            for desired_velocity, waypoint in zip(self.desired_velocities, self.waypoints_list):
                waypoint.velocity = desired_velocity
            
        
        # 
        # calculate closest index
        # 
        closest_relative_index, self.closest_distance = WarthogEnv.get_closest(
            remaining_waypoints=self.waypoints_list[self.next_waypoint_index:],
            x=self.spacial_info.x,
            y=self.spacial_info.y,
        )
        self.next_waypoint_index += closest_relative_index
        self.prev_next_waypoint_index = self.next_waypoint_index
        
        self.observation = WarthogEnv.generate_observation(
            closest_index=self.next_waypoint_index,
            remaining_waypoints=self.waypoints_list[self.next_waypoint_index:],
            current_spacial_info=self.spacial_info,
        )
        
        self.renderer = Renderer(
            vehicle_render_width=config.vehicle.render_width,
            vehicle_render_length=config.vehicle.render_length,
            waypoints_list=self.waypoints_list,
            should_render=config.simulator.should_render and countdown(config.simulator.render_rate),
            inital_x=self.spacial_info.x,
            inital_y=self.spacial_info.y,
            render_axis_size=20,
            render_path=f"{config.output_folder}/render/",
            history_size=config.simulator.number_of_trajectories,
        )
        
        return self.observation

    @property
    def number_of_waypoints(self):
        return len(self.waypoints_list)
    