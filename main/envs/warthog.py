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

from config import config, path_to, grug_test, debug
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

def advance_the_index_if_needed(remaining_waypoints,x,y):
    closest_distance = math.inf
    for i, each in enumerate(remaining_waypoints):
        dist = get_distance(x1=each[0], y1=each[1], x2=x, y2=y)
        if dist <= closest_distance:
            closest_distance = dist
            change_in_waypoint_index = i
        else:
            break
    
    return  change_in_waypoint_index, closest_distance

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
        # args
        if True:
            self.waypoint_file_path     = waypoint_file_path
            self.trajectory_output_path = trajectory_output_path
            self.recorder               = recorder
        
        # episode-independent non-constants 
        if True:
            self.global_timestep = 0
        
        # file based data
        if True:
            self.desired_velocities, self.waypoints_list = read_waypoint_file(self.waypoint_file_path)
            for desired_velocity, waypoint in zip(self.desired_velocities, self.waypoints_list):
                waypoint.velocity = desired_velocity
            
            # 
            # trajectory_file
            # 
            self.trajectory_file = None
            if self.trajectory_output_path is not None:
                print(f'''trajectory being logged to: {trajectory_output_path}''')
                FS.ensure_is_folder(FS.parent_path(trajectory_output_path))
                self.trajectory_file = open(trajectory_output_path, "w+")
                self.trajectory_file.writelines(f"x, y, angle, velocity, spin, velocity_action, spin_action, is_episode_start\n")
        
    def __del__(self):
        if self.trajectory_file:
            self.trajectory_file.close()
    
    @staticmethod
    @grug_test(max_io=30, record_io=None, additional_io_per_run=None, skip=True)
    def generate_observation(remaining_waypoints, current_spacial_info):
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
    def generate_next_spacial_info(old_spacial_info, action_relative, action_duration, debug=False, **kwargs):
        '''
            Note:
                This function should also be fully deterministic
        '''
        
        def one_step(action_absolute, prev_spacial_info, action_duration):
            """
            # original
                def sim_warthog(self, v, w):
                    x = self.pose[0]
                    y = self.pose[1]
                    th = self.pose[2]
                    v_ = self.twist[0]
                    w_ = self.twist[1]
                    self.twist[0] = v
                    self.twist[1] = w
                    if self.is_delayed_dynamics:
                        v_ = self.v_delay_data[0]
                        w_ = self.w_delay_data[0]
                        del self.v_delay_data[0]
                        del self.w_delay_data[0]
                        self.v_delay_data.append(v)
                        self.w_delay_data.append(w)
                        self.twist[0] = self.v_delay_data[0]
                        self.twist[1] = self.v_delay_data[1]
                    dt = self.dt
                    self.prev_ang = self.pose[2]
                    self.pose[0] = x + v_ * math.cos(th) * dt
                    self.pose[1] = y + v_ * math.sin(th) * dt
                    self.pose[2] = th + w_ * dt
                    self.ep_poses.append(np.array([x, y, th, v_, w_, v, w]))
                    self.ep_start = 0
                
                # aka:
                def sim_warthog(v, w, pose, twist, dt):
                    x,y,th = pose
                    v_,w_,*_ = twist
                    twist[0] = v
                    twist[1] = w
                    pose[0] = x + v_ * math.cos(th) * dt
                    pose[1] = y + v_ * math.sin(th) * dt
                    pose[2] = th + w_ * dt
            """
            return SpacialInformation(
                x=prev_spacial_info.x + prev_spacial_info.velocity * math.cos(prev_spacial_info.angle) * action_duration,
                y=prev_spacial_info.y + prev_spacial_info.velocity * math.sin(prev_spacial_info.angle) * action_duration,
                angle=prev_spacial_info.angle + prev_spacial_info.spin * action_duration,
                velocity=action_absolute.velocity,
                spin=action_absolute.spin,
                timestep=-1,
            )
        
        action_absolute = Action(
            velocity=action_relative.velocity * config.vehicle.controller_max_velocity,
            spin=action_relative.spin * config.vehicle.controller_max_spin,
        )
        
        running_spacial_info = old_spacial_info
        for each in range(config.simulator.granularity_of_calculations):
            running_spacial_info = one_step(
                action_absolute=action_absolute,
                prev_spacial_info=running_spacial_info,
                action_duration=action_duration/config.simulator.granularity_of_calculations,
            )
        
        return running_spacial_info
    
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
    def original_reward_function(*, spacial_info, closest_distance, relative_action, prev_relative_action, closest_waypoint, closest_relative_index,):
        relative_velocity      = relative_action.velocity
        relative_spin          = relative_action.spin
        prev_relative_velocity = prev_relative_action.velocity
        prev_relative_spin     = prev_relative_action.spin
        
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
    
    def reaction(
        self,
        action_buffer,
        with_delay=0,
        with_additive_adversity=False,
        with_battery_adversity=False,
        with_gaussian_noise=False,
        clipped=False,
        absolute_units=False
    ):
        """
        Needs:
            1. self.action_buffer needs to contain the most recent action as the first element
               and needs to contain relative-original actions
               and the buffer needs to be updated each .step()
            2. self.simulated_battery_level to be changed each .step()
            3. self.random_seed_table_for_actions just needs to be available
        """
        # check
        if len(action_buffer)-1 < with_delay:
            raise Exception(f'''\n\nRe-run the code with `action_delay` in config.yaml set to {with_delay}''')
        
        original_action = action_buffer[-(with_delay+1)]
        output_action = original_action
        
        if with_additive_adversity:
            output_action = Action(
                # clipping is mostly to ensure the velocity never goes negative
                velocity=output_action.velocity + config.simulator.velocity_offset,
                spin=output_action.spin + config.simulator.spin_offset,
            )
        
        if with_battery_adversity:
            output_action = Action(
                # clipping is mostly to ensure the velocity never goes negative
                velocity=clip(output_action.velocity*self.simulated_battery_level, min=0),
                spin=output_action.spin,
            )
        
        if with_gaussian_noise:
            noise = None
            if original_action in self.random_seed_table_for_actions:
                noise = self.random_seed_table_for_actions[original_action]
            else:
                noise = self.random_seed_table_for_actions[original_action] = Action(
                    velocity=random.normalvariate(mu=0, sigma=config.simulator.gaussian_action_noise.velocity_action.standard_deviation, ),
                    spin=random.normalvariate(mu=0, sigma=config.simulator.gaussian_action_noise.spin_action.standard_deviation, ),
                )
            
            output_action = Action(
                velocity=output_action.velocity + noise.velocity,
                spin=output_action.spin + noise.spin,
            )
        
        if clipped:
            output_action = Action(
                velocity=clip(output_action.velocity, min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity),
                spin=clip(output_action.spin        , min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    ),
            )
        
        if absolute_units:
            output_action = Action(
                velocity=output_action.velocity * config.vehicle.controller_max_velocity,
                spin=output_action.spin * config.vehicle.controller_max_spin,
            )
        
        return output_action
    
    def step(self, action, override_next_spacial_info=None):
        """
            Note:
                this is where all the noise (action noise, observation noise) is added.
                Note: noise is added before the values are scaled, which might amplify noise.
                
                `override_next_spacial_info` is currently only used with ROS runtime. This
                allows for most of the code to stay as-is while throwing away simulated data
                in favor of real-world data.
        """
        step_data = LazyDict()
        # 
        # logging
        # 
        if config.simulator.save_data:
            if self.trajectory_file is not None:
                self.trajectory_file.writelines(f"{self.spacial_info.x}, {self.spacial_info.y}, {self.spacial_info.angle}, {self.spacial_info.velocity}, {self.spacial_info.spin}, {self.action_buffer[0].velocity}, {self.action_buffer[0].spin}, {self.is_episode_start}\n")
            
            if config.simulator.battery_adversity_enabled:
                self.recorder.add(timestep=self.global_timestep, simulated_battery_level=self.simulated_battery_level)
                self.recorder.commit()
        #  
        # increment things
        # 
        if True:
            # timestep 
            self.global_timestep  += 1
            self.episode_timestep += 1
            self.is_episode_start = False
            
            # battery level
            self.simulated_battery_level *= 1-config.simulator.battery_decay_rate
            
            # action
            self.action_buffer.append( Action(velocity=action[0], spin=action[1]) )
            self.action_buffer = self.action_buffer[-max(config.simulator.action_delay,2):] # 2 is so that we effectively have prev_action even when delay = 0
            # cap the size (remove oldest), but never let size fall below 1 (should always contain the most-recent desired action)
        
        # 
        # modify action
        # 
        if True:
            mutated_action_relative = self.reaction(
                self.action_buffer,
                with_delay=config.simulator.action_delay,
                with_additive_adversity=config.simulator.additive_adversity_enabled,
                with_battery_adversity=config.simulator.battery_adversity_enabled,
                with_gaussian_noise=config.simulator.use_gaussian_action_noise,
                clipped=True,
                absolute_units=False,
            )
        
        # 
        # modify spacial_info
        # 
        self.prev_spacial_info = self.spacial_info
        self.spacial_info = (
            override_next_spacial_info
                if type(override_next_spacial_info) != type(None)
                    else WarthogEnv.generate_next_spacial_info(
                        old_spacial_info=SpacialInformation(*self.spacial_info),
                        action_relative=mutated_action_relative,
                        action_duration=config.simulator.action_duration,
                    )
        )
        # 
        # add spacial noise
        #
        self.prev_spacial_info_with_noise = self.spacial_info_with_noise
        self.spacial_info_with_noise = self.spacial_info
        if config.simulator.use_gaussian_spacial_noise:
            self.spacial_info_with_noise = SpacialInformation(
                self.spacial_info.x        + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.x.standard_deviation       , ),
                self.spacial_info.y        + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.y.standard_deviation       , ),
                self.spacial_info.angle    + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.angle.standard_deviation   , ),
                self.spacial_info.velocity + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.spin.standard_deviation    , ),
                self.spacial_info.spin     + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.velocity.standard_deviation, ),
                self.spacial_info.timestep ,
            )
        
        # 
        # increment waypoints
        # 
        # FIXME: make it so that waypoints can't be skipped (e.g. shortcuts)
        change_in_waypoint_index, step_data.closest_distance = advance_the_index_if_needed(
            remaining_waypoints=self.waypoints_list[self.next_waypoint_index:],
            x=self.spacial_info.x,
            y=self.spacial_info.y,
        )
        self.prev_next_waypoint_index = self.next_waypoint_index
        self.next_waypoint_index += change_in_waypoint_index
        
        # 
        # Reward Calculation
        # 
        step_data.reward, step_data.velocity_error, step_data.crosstrack_error, step_data.phi_error = WarthogEnv.almost_original_reward_function(
            spacial_info=self.spacial_info,
            closest_distance=step_data.closest_distance,
            relative_action=mutated_action_relative,
            prev_relative_action=self.reaction(
                self.action_buffer,
                with_delay=1, # this is what makes it "previous"
                with_additive_adversity=config.simulator.additive_adversity_enabled,
                with_battery_adversity=config.simulator.battery_adversity_enabled,
                with_gaussian_noise=config.simulator.use_gaussian_action_noise,
                clipped=True,
                absolute_units=False,
            ),
            closest_waypoint=self.waypoints_list[self.next_waypoint_index],
            closest_relative_index=1 if change_in_waypoint_index > 0 else 0,
        )
        self.total_episode_reward += step_data.reward
        
        # 
        # make mutated observation
        # 
        self.prev_observation = self.observation
        self.observation = WarthogEnv.generate_observation(
            remaining_waypoints=self.waypoints_list[self.next_waypoint_index:],
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
            text_data=f"vel_error={step_data.velocity_error:.3f}\nclosest_index={self.next_waypoint_index}\ncrosstrack_error={step_data.crosstrack_error:.3f}\nReward={step_data.reward:.4f}\nwarthog_vel={self.spacial_info.velocity:.3f}\nphi_error={step_data.phi_error*180/math.pi:.4f}\nep_reward={self.total_episode_reward:.4f}\n\nomega_reward={(-2 * math.fabs(self.action_buffer[-1].spin)):.4f}\nvel_reward={step_data.velocity_error:.4f}",
        )
        
        additional_info = AdditionalInfo(
            timestep_index=Unknown,
            action_duration=config.simulator.action_duration,
            spacial_info=self.prev_spacial_info,
            spacial_info_with_noise=self.prev_spacial_info_with_noise,
            observation_from_spacial_info_with_noise=self.prev_observation,
            historic_transform=Unknown,
            original_reaction=self.action_buffer[-1],
            mutated_relative_reaction=mutated_action_relative,
            next_spacial_info=self.spacial_info,
            next_spacial_info_spacial_info_with_noise=self.spacial_info_with_noise,
            next_observation_from_spacial_info_with_noise=self.observation,
            next_closest_index=self.next_waypoint_index,
            reward=step_data.reward,
        )
        
        # 
        # done Calculation
        #
        step_data.done = False
        if self.next_waypoint_index >= len(self.waypoints_list) - 1:
            step_data.done = True 
        if self.episode_timestep == config.simulator.max_number_of_timesteps_per_episode:
            step_data.done = True
            self.episode_timestep = 0
        # immediate end due to too much loss
        if config.simulator.allow_cut_short_episode:
            if math.fabs(step_data.crosstrack_error) > magic_number_1_point_5 or math.fabs(step_data.phi_error) > magic_number_1_point_4:
                step_data.done = True
        
        return self.observation, step_data.reward, step_data.done, additional_info

    def reset(self, override_next_spacial_info=None):
        assert len(self.waypoints_list) > config.simulator.min_number_of_remaining_waypoints, f"There's less than {config.simulator.min_number_of_remaining_waypoints} waypoints, and I think that indicates a problem"
        
        # 
        # NOTE: these are effectively in chronological order with a single-assignment for each var to make it easier to think about
        # 
        self.prev_spacial_info                   = SpacialInformation(x=0, y=0, angle=0, velocity=0, spin=0, timestep=-1)
        self.prev_spacial_info_with_noise        = SpacialInformation(x=0, y=0, angle=0, velocity=0, spin=0, timestep=-1)
        self.prev_next_waypoint_index            = (
            config.simulator.starting_waypoint
                if config.simulator.starting_waypoint != 'random'
                else
                    np.random.randint(len(self.waypoints_list)-(config.simulator.min_number_of_remaining_waypoints-1), size=1)[0]
        )
        self.prev_observation                    = None
        self.episode_timestep                    = 0
        self.is_episode_start                    = True
        self.spacial_info = (
            override_next_spacial_info # if position is overriden by (most likely) the real world position
                if type(override_next_spacial_info) != type(None)
                else
                    SpacialInformation(
                        x=float(self.waypoints_list[self.prev_next_waypoint_index][0] + config.simulator.random_start_position_offset),
                        y=float(self.waypoints_list[self.prev_next_waypoint_index][1] + config.simulator.random_start_position_offset),
                        angle=float(self.waypoints_list[self.prev_next_waypoint_index][2] + config.simulator.random_start_angle_offset),
                        velocity=0,
                        spin=0,
                        timestep=0,
                    )
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
        self.spacial_info_with_noise = self.spacial_info
        closest_relative_index, _closest_distance = WarthogEnv.get_closest(
            remaining_waypoints=self.waypoints_list[self.prev_next_waypoint_index:],
            x=self.spacial_info.x,
            y=self.spacial_info.y,
        )
        self.next_waypoint_index                 = self.prev_next_waypoint_index + closest_relative_index # often closest_relative_index==0
        self.observation = WarthogEnv.generate_observation(
            remaining_waypoints=self.waypoints_list[self.next_waypoint_index:],
            current_spacial_info=self.spacial_info,
        )
        self.action_buffer                       = [ Action(velocity=0,spin=0) ] * (config.simulator.action_delay+1) # seed so that prev_action effectively works
        self.simulated_battery_level             = 1.0 # proportion 
        self.random_seed_table_for_actions       = {}
        self.total_episode_reward                = 0
        
        return self.observation