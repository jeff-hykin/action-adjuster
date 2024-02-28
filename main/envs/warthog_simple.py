import math
import time
import csv
from copy import deepcopy
from collections import namedtuple
import json
import random

import gym
import numpy as np
from config import grug_test, path_to, config
from envs.warthog import read_waypoint_file, SpacialInformation


from generic_tools.geometry import zero_to_2pi, pi_to_pi
from __dependencies__.blissful_basics import Csv, create_named_list_class, FS, print, stringify, clip, countdown, LazyDict

from config import config, path_to, grug_test
from generic_tools.geometry import get_distance, get_angle_from_origin, zero_to_2pi, pi_to_pi, abs_angle_difference, angle_created_by

from data_structures import Unknown, Action, StepOutput, StepSideEffects, GetObservationOutput, RewardOutput, SimWarthogOutput, PoseEntry, TwistEntry, SpacialHistory, SpacialInformation, ReactionClass, WaypointGap, Waypoint, Observation, AdditionalInfo
from render import Renderer
from misc import are_equal, scaled_sigmoid, no_duplicates

def pose_and_twist_to_spacial(pose, twist):
    return SpacialInformation(
        x=pose.x,
        y=pose.y,
        angle=pose.angle,
        velocity=twist.velocity,
        spin=twist.spin,
        timestep=-1,
    )

def next_spacial_info(absolute_action, prev_spacial_info, action_duration):
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
        velocity=absolute_action.velocity,
        spin=absolute_action.spin,
        timestep=-1,
    )

global_a_next_spacial_buffer = []
global_b_next_spacial_buffer = []
@grug_test(max_io=5, skip=False)
def generate_next_spacial_info(
    old_spacial_info,
    relative_velocity,
    relative_spin,
    action_duration,
    ep_poses,
    **kwargs,
):
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
    absolute_action = Action(
        np.clip(relative_velocity, 0, 1) * 4.0,
        np.clip(relative_spin, -1, 1)    * 2.5,
    )
    running_spacial_info = old_spacial_info
    for each in range(config.simulator.granularity_of_calculations):
        running_spacial_info = next_spacial_info(
            absolute_action=Action(velocity=absolute_action.velocity, spin=absolute_action.spin),
            prev_spacial_info=running_spacial_info,
            # NOTE: why the *4? I have no idea. It shouldn't be needed but the driving is slow without it
            #       Even the original one (no loops) drove too slow
            #       HOWEVER doing *4 on the original one causes problems for some reason
            #       TODO / FIXME
            action_duration=4*(action_duration/config.simulator.granularity_of_calculations),
        )
    return running_spacial_info
    
    effective_action_duration = action_duration/config.simulator.granularity_of_calculations
    pose = PoseEntry(
        x=old_spacial_info.x,
        y=old_spacial_info.y,
        angle=old_spacial_info.angle,
    )
    twist = TwistEntry(
        velocity=old_spacial_info.velocity,
        spin=old_spacial_info.spin,
        unknown=None,
    )
    absolute_velocity = absolute_action.velocity
    absolute_spin = absolute_action.spin
    original_one_shot = sim_warthog(
        v=absolute_velocity,
        w=absolute_spin,
        dt=action_duration,
        pose=PoseEntry(
            x=old_spacial_info.x,
            y=old_spacial_info.y,
            angle=old_spacial_info.angle,
        ),
        twist=TwistEntry(
            velocity=old_spacial_info.velocity,
            spin=old_spacial_info.spin,
            unknown=None,
        ),
    )
    original_one_shot_2 = sim_warthog_2(
        action=absolute_action,
        dt=action_duration,
        spacial_info=old_spacial_info,
    )
    original_two_step = sim_warthog_2(
        action=absolute_action,
        dt=action_duration,
        spacial_info=original_one_shot_2,
    )
    
    
    old_v, old_w, *unknown = twist
    twist = TwistEntry(
        velocity=float(absolute_velocity),
        spin=float(absolute_spin),
        unknown=(None if len(unknown) == 0 else unknown[0]),
    )
    one_shot_position = pose_and_twist_to_spacial(
        pose=PoseEntry(
            x=float(pose.x + twist.velocity * math.cos(pose.angle) * action_duration),
            y=float(pose.y + twist.velocity * math.sin(pose.angle) * action_duration),
            angle=float(zero_to_2pi(pose.angle + twist.spin * action_duration)),
        ),
        twist=twist,
    )
    old_v, old_w, *unknown = twist
    for each in range(config.simulator.granularity_of_calculations*4):
        old_x, old_y, prev_angle = pose
        pose = PoseEntry(
            x=float(old_x + old_v * math.cos(prev_angle) * effective_action_duration),
            y=float(old_y + old_v * math.sin(prev_angle) * effective_action_duration),
            angle=float(zero_to_2pi(prev_angle + old_w * effective_action_duration)),
        )
        global_a_next_spacial_buffer.append(pose_and_twist_to_spacial(pose, twist))
    
    print(f'''generate_next_spacial_info''')
    import code; code.interact(banner='',local={**globals(),**locals()})
    ep_poses.append(
        SpacialHistory(
            x=float(old_x),
            y=float(old_y),
            angle=float(prev_angle),
            velocity=float(old_v),
            spin=float(old_w),
            new_velocity=float(absolute_velocity),
            new_spin=float(absolute_spin),
        )
    )
    return SimWarthogOutput(twist, prev_angle, pose, ep_poses, absolute_action)

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


@grug_test(max_io=5, skip=False)
def pure_get_observation(
    next_waypoint_index_,
    horizon,
    number_of_waypoints,
    pose,
    twist,
    waypoints_list,
    **kwargs,
):
    change_in_waypoint_index, closest_distance = advance_the_index_if_needed(
        remaining_waypoints=waypoints_list[next_waypoint_index_:],
        x=pose[0],
        y=pose[1],
    )
    next_waypoint_index_ += change_in_waypoint_index
    
    obs = SimpleHelpers.generate_observation_array(
        next_waypoint_index_,
        horizon,
        number_of_waypoints,
        pose,
        twist,
        waypoints_list,
    )
    return GetObservationOutput(obs, closest_distance, next_waypoint_index_)

@grug_test(max_io=5, skip=False)
def pure_reward(
    closest_waypoint,
    pose,
    twist,
    closest_distance,
    action,
    prev_absolute_action,
    **kwargs,
):
    waypoint_x, waypoint_y, waypoint_phi, waypoint_velocity, *_ = closest_waypoint
    pose_x, pose_y, pose_phi, *_ = pose
    
    x_diff = waypoint_x - pose_x
    y_diff = waypoint_y - pose_y
    yaw_error = pi_to_pi(get_angle_from_origin(x_diff, y_diff) - pose_phi)
    phi_error = pi_to_pi(
        waypoint_phi - pose_phi
    )
    velocity_error = waypoint_velocity - twist[0]
    crosstrack_error = closest_distance * math.sin(yaw_error)
    reward = (
        (2.0 - math.fabs(crosstrack_error))
        * (4.5 - math.fabs(velocity_error))
        * (math.pi / 3.0 - math.fabs(phi_error))
        - math.fabs(action[0] - prev_absolute_action[0])
        - 2 * math.fabs(action[1])
    )
    if waypoint_velocity >= 2.5 and math.fabs(velocity_error) > 1.5:
        reward = 0
    elif waypoint_velocity < 2.5 and math.fabs(velocity_error) > 0.5:
        reward = 0
    
    return RewardOutput(reward, velocity_error, crosstrack_error, phi_error)

@grug_test(max_io=5, skip=False)
def pure_reward_wrapper(
    total_episode_reward,
    next_waypoint_index_,
    waypoints_list,
    pose,
    twist,
    closest_distance,
    episode_steps,
    max_number_of_timesteps_per_episode,
    action,
    prev_absolute_action,
    done,
    **kwargs,
):
    xdiff         = waypoints_list[next_waypoint_index_][0] - pose[0]
    ydiff         = waypoints_list[next_waypoint_index_][1] - pose[1]
    yaw_error     = pi_to_pi(get_angle_from_origin(xdiff, ydiff) - pose[2])
    omega_reward  = -2 * math.fabs(action[1])
    velocity_reward    = -math.fabs(action[0] - prev_absolute_action[0])
    
    reward, velocity_error, crosstrack_error, phi_error = pure_reward(
        closest_waypoint=waypoints_list[next_waypoint_index_],
        pose=pose,
        twist=twist,
        closest_distance=closest_distance,
        action=action,
        prev_absolute_action=prev_absolute_action,
    )
    
    if math.fabs(crosstrack_error) > 1.5 or math.fabs(phi_error) > 1.4:
        done = True
    
    if episode_steps >= max_number_of_timesteps_per_episode:
        done = True
        episode_steps = 0
    
    total_episode_reward = total_episode_reward + reward
    prev_absolute_action = action
    
    return reward, crosstrack_error, xdiff, ydiff, yaw_error, phi_error, velocity_error, done, episode_steps, omega_reward, velocity_reward, prev_absolute_action, total_episode_reward


@grug_test(max_io=10, skip=False)
def pure_step(
    relative_action,
    absolute_action,
    next_waypoint_index_,
    crosstrack_error,
    action_duration,
    ep_poses,
    episode_steps,
    horizon,
    max_number_of_timesteps_per_episode,
    number_of_waypoints,
    omega_reward,
    phi_error,
    pose,
    prev_absolute_action,
    prev_angle,
    reward,
    total_episode_reward,
    twist,
    velocity_error,
    velocity_reward,
    waypoints_list,
    **kwargs,
):
    prev_next_waypoint_index_ = next_waypoint_index_
    obs, closest_distance, next_waypoint_index_ = pure_get_observation(
        next_waypoint_index_=next_waypoint_index_,
        horizon=horizon,
        number_of_waypoints=number_of_waypoints,
        pose=pose,
        twist=twist,
        waypoints_list=waypoints_list,
    )
    done = False
    if next_waypoint_index_ >= number_of_waypoints - 1:
        done = True
    
    reward, crosstrack_error, xdiff, ydiff, yaw_error, phi_error, velocity_error, done, episode_steps, omega_reward, velocity_reward, prev_absolute_action, total_episode_reward = pure_reward_wrapper(
        total_episode_reward=total_episode_reward,
        next_waypoint_index_=next_waypoint_index_,
        waypoints_list=waypoints_list,
        pose=pose,
        twist=twist,
        closest_distance=closest_distance,
        episode_steps=episode_steps,
        max_number_of_timesteps_per_episode=max_number_of_timesteps_per_episode,
        action=absolute_action,
        prev_absolute_action=prev_absolute_action,
        done=done,
    )
    
    return (
        StepOutput(
            obs,
            reward,
            done,
            {}
        ),
        StepSideEffects(
            absolute_action,
            crosstrack_error,
            episode_steps,
            omega_reward,
            phi_error,
            prev_absolute_action,
            prev_next_waypoint_index_,
            reward,
            total_episode_reward,
            velocity_error,
            velocity_reward,
            twist,
            prev_angle,
            pose,
            closest_distance,
            next_waypoint_index_,
            ep_poses,
        ),
        (
            yaw_error,
            ydiff,
            xdiff,
        )
    )

magic_number_1_point_5 = 1.5
magic_number_1_point_4 = 1.4

class WarthogEnv(gym.Env):
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
    
    def __init__(self, waypoint_file_path, trajectory_output_path=None, recorder=None, *args, **kwargs):
        super(WarthogEnv, self).__init__()
        # 
        # A
        # 
        self.a = LazyDict()
        if True:
            self.a.waypoint_file_path = waypoint_file_path
            self.a.trajectory_output_path = trajectory_output_path
            self.a.recorder = recorder
            
            self.a.waypoints_list   = []
            self.a.prev_spacial_info = SpacialInformation(0,0,0,0,0,-1)
            self.a.prev_spacial_info_with_noise = SpacialInformation(0,0,0,0,0,-1)
            self.a.spacial_info_with_noise = SpacialInformation(0,0,0,0,0,0)
            self.a.spacial_info = SpacialInformation(
                x                = 0,
                y                = 0,
                angle            = 0,
                velocity         = 0,
                spin             = 0,
                timestep         = 0,
            )
            self.a.observation = None
            
            self.a.max_number_of_timesteps_per_episode      = config.simulator.max_number_of_timesteps_per_episode
            self.a.save_data              = config.simulator.save_data
            self.a.action_duration        = config.simulator.action_duration  
            self.a.next_waypoint_index_new   = 0
            self.a.prev_next_waypoint_index  = 0
            self.a.closest_distance           = math.inf
            self.a.desired_velocities         = []
            self.a.episode_steps              = 0
            self.a.total_episode_reward       = 0
            self.a.reward                     = 0
            self.a.original_relative_spin           = 0 
            self.a.original_relative_velocity       = 0 
            self.a.prev_original_relative_spin      = 0 
            self.a.prev_original_relative_velocity  = 0 # "original" is what the actor said to do
            self.a.mutated_relative_spin            = 0 # "mutated" is after adversity+noise was added
            self.a.mutated_relative_velocity        = 0
            self.a.prev_mutated_relative_spin       = 0
            self.a.prev_mutated_relative_velocity   = 0
            self.a.is_episode_start        = 1
            self.a.trajectory_file         = None
            self.a.global_timestep         = 0
            self.a.action_buffer           = [ (0,0) ] * config.simulator.action_delay # seed the buffer with delays
            self.a.simulated_battery_level = 1.0 # proportion 
            
            if self.a.waypoint_file_path is not None:
                self.a.desired_velocities, self.a.waypoints_list = read_waypoint_file(self.a.waypoint_file_path)
            
            self.a.crosstrack_error = 0
            self.a.velocity_error   = 0
            self.a.phi_error        = 0
            
            self.a.waypoints_list = []
            self.a.pose = PoseEntry(
                x=0,
                y=0,
                angle=0,
            )
            self.a.twist = TwistEntry(
                velocity=0,
                spin=0,
                unknown=0,
            )
            self.a.next_waypoint_index = 0
            self.a.prev_next_waypoint_index = 0
            self.a.closest_distance = math.inf
            self.a.horizon = 10
            self.a.action_duration = 0.06
            self.a.desired_velocities, self.a.waypoints_list = read_waypoint_file(waypoint_file_path)
            self.a.number_of_waypoints = len(self.a.waypoints_list)
            self.a.max_vel = 1
            self.a.prev_angle              = 0
            self.a.crosstrack_error        = 0
            self.a.velocity_error               = 0
            self.a.phi_error               = 0
            self.a.max_number_of_timesteps_per_episode = 700
            self.a.total_episode_reward    = 0
            self.a.reward                  = 0
            self.a.absolute_action         = [0.0, 0.0]
            self.a.prev_absolute_action    = [0.0, 0.0]
            self.a.velocity_reward         = 0
            self.a.save_data               = config.simulator.save_data
            self.a.ep_poses                = []
            
            self.a.global_timestep = 0
            self.a.original_relative_spin = 0 
        
        # 
        # B
        # 
        self.b = LazyDict()
        if True:
            self.b.waypoint_file_path = waypoint_file_path
            self.b.trajectory_output_path = trajectory_output_path
            self.b.recorder = recorder
            
            self.b.waypoints_list   = []
            self.b.prev_spacial_info = SpacialInformation(0,0,0,0,0,-1)
            self.b.prev_spacial_info_with_noise = SpacialInformation(0,0,0,0,0,-1)
            self.b.spacial_info_with_noise = SpacialInformation(0,0,0,0,0,0)
            self.b.spacial_info = SpacialInformation(
                x                = 0,
                y                = 0,
                angle            = 0,
                velocity         = 0,
                spin             = 0,
                timestep         = 0,
            )
            self.b.observation = None
            
            self.b.max_number_of_timesteps_per_episode      = config.simulator.max_number_of_timesteps_per_episode
            self.b.save_data              = config.simulator.save_data
            self.b.action_duration        = config.simulator.action_duration  
            self.b.next_waypoint_index    = 0
            self.b.prev_next_waypoint_index     = 0
            self.b.closest_distance       = math.inf
            self.b.desired_velocities     = []
            self.b.episode_steps          = 0
            self.b.total_episode_reward   = 0
            self.b.reward                 = 0
            self.b.original_relative_spin           = 0 
            self.b.original_relative_velocity       = 0 
            self.b.prev_original_relative_spin      = 0 
            self.b.prev_original_relative_velocity  = 0 # "original" is what the actor said to do
            self.b.mutated_relative_spin            = 0 # "mutated" is after adversity+noise was added
            self.b.mutated_relative_velocity        = 0
            self.b.prev_mutated_relative_spin       = 0
            self.b.prev_mutated_relative_velocity   = 0
            self.b.observation        = None
            self.b.is_episode_start        = 1
            self.b.trajectory_file         = None
            self.b.global_timestep         = 0
            self.b.action_buffer           = [ (0,0) ] * config.simulator.action_delay # seed the buffer with delays
            self.b.simulated_battery_level = 1.0 # proportion 
            
            if self.b.waypoint_file_path is not None:
                self.b.desired_velocities, self.b.waypoints_list = read_waypoint_file(self.b.waypoint_file_path)
            
            self.b.crosstrack_error = 0
            self.b.velocity_error   = 0
            self.b.phi_error        = 0
            
            # 
            # trajectory_file
            # 
            if self.b.trajectory_output_path is not None:
                # print(f'''trajectory being logged to: {trajectory_output_path}''')
                # FS.ensure_is_folder(FS.parent_path(trajectory_output_path))
                self.b.trajectory_file = open(trajectory_output_path, "w+")
                self.b.trajectory_file.writelines(f"x, y, angle, velocity, spin, velocity_action, spin_action, is_episode_start\n")
        
        # 
        # C
        # 
        self.c = LazyDict()
        if True:
            self.c.waypoint_file_path = waypoint_file_path
            self.c.trajectory_output_path = trajectory_output_path
            self.c.recorder = recorder
            
            self.c.prev_spacial_info = SpacialInformation(0,0,0,0,0,-1)
            self.c.prev_spacial_info_with_noise = SpacialInformation(0,0,0,0,0,-1)
            self.c.spacial_info_with_noise = SpacialInformation(0,0,0,0,0,0)
            self.c.spacial_info = SpacialInformation(
                x                = 0,
                y                = 0,
                angle            = 0,
                velocity         = 0,
                spin             = 0,
                timestep         = 0,
            )
            self.c.observation = None
            
            self.c.max_number_of_timesteps_per_episode = config.simulator.max_number_of_timesteps_per_episode
            self.c.save_data                           = config.simulator.save_data
            self.c.action_duration                     = config.simulator.action_duration  
            self.c.next_waypoint_index                 = 0
            self.c.prev_next_waypoint_index            = 0
            self.c.closest_distance                    = math.inf
            self.c.episode_steps                       = 0
            self.c.total_episode_reward                = 0
            self.c.reward                              = 0
            self.c.original_relative_spin           = 0 
            self.c.original_relative_velocity       = 0 
            self.c.prev_original_relative_spin      = 0 
            self.c.prev_original_relative_velocity  = 0 # "original" is what the actor said to do
            self.c.mutated_relative_spin            = 0 # "mutated" is after adversity+noise was added
            self.c.mutated_relative_velocity        = 0
            self.c.prev_mutated_relative_spin       = 0
            self.c.prev_mutated_relative_velocity   = 0
            self.c.is_episode_start        = 1
            self.c.trajectory_file         = None
            self.c.global_timestep         = 0
            self.c.action_buffer           = [ (0,0) ] * config.simulator.action_delay # seed the buffer with delays
            self.c.simulated_battery_level = 1.0 # proportion 
            
            if self.c.waypoint_file_path is not None:
                self.c.desired_velocities, self.c.waypoints_list = read_waypoint_file(self.c.waypoint_file_path)
            
            self.c.crosstrack_error = 0
            self.c.velocity_error   = 0
            self.c.phi_error        = 0
            
            # 
            # trajectory_file
            # 
            if self.c.trajectory_output_path is not None:
                # print(f'''trajectory being logged to: {trajectory_output_path}''')
                # FS.ensure_is_folder(FS.parent_path(trajectory_output_path))
                self.c.trajectory_file = open(trajectory_output_path, "w+")
                self.c.trajectory_file.writelines(f"x, y, angle, velocity, spin, velocity_action, spin_action, is_episode_start\n")
            
            self.c.pose = PoseEntry(
                x=0,
                y=0,
                angle=0,
            )
            self.c.twist = TwistEntry(
                velocity=0,
                spin=0,
                unknown=0,
            )
            self.c.max_vel = 1
            self.c.prev_angle              = 0
            self.c.original_action         = [0.0, 0.0]
            self.c.absolute_action         = [0.0, 0.0]
            self.c.prev_absolute_action    = [0.0, 0.0]
            self.c.velocity_reward         = 0
            self.c.ep_poses                = []
            
            self.c.global_timestep = 0

        self.reset()
        # self.diff_compare(print_c=True)
    
    def __del__(self):
        if self.a.trajectory_file:
            self.a.trajectory_file.close()
    
    @property
    def _(self):
        return LazyDict({
            key: getattr(self.a, key, getattr(self.b, key, None)) for key in [
                "waypoint_file_path",
                "trajectory_output_path",
                "recorder",
                "waypoints_list",
                "prev_spacial_info",
                "prev_spacial_info_with_noise",
                "spacial_info_with_noise",
                "spacial_info",
                "observation",
                "max_number_of_timesteps_per_episode",
                "save_data",
                "action_duration",
                "next_waypoint_index",
                "prev_next_waypoint_index_",
                "closest_distance",
                "desired_velocities",
                "episode_steps",
                "total_episode_reward",
                "reward",
                "original_relative_spin",
                "original_relative_velocity",
                "prev_original_relative_spin",
                "prev_original_relative_velocity",
                "mutated_relative_spin",
                "mutated_relative_velocity",
                "prev_mutated_relative_spin",
                "prev_mutated_relative_velocity",
                "prev_observation",
                "is_episode_start",
                "trajectory_file",
                "global_timestep",
                "action_buffer",
                "simulated_battery_level",
                "crosstrack_error",
                "velocity_error",
                "phi_error",
                "pose",
                "twist",
                "next_waypoint_index_",
                "horizon",
                "number_of_waypoints",
                "max_vel",
                "prev_angle",
                "action",
                "absolute_action",
                "prev_absolute_action",
                "velocity_reward",
                "ep_poses",
                "key",
                "relative_action",
            ]
        })
    
    # just a wrapper around the pure_step
    def step(self, action, override_next_spacial_info=None):
        # 
        # part1
        # 
        if True:
            # 
            # A
            # 
            if True:
                a = LazyDict()
                #  
                # push new action
                # 
                self.a.prev_original_relative_velocity = self.a.original_relative_velocity
                self.a.prev_original_relative_spin     = self.a.original_relative_spin
                self.a.prev_mutated_relative_velocity  = self.a.mutated_relative_velocity
                self.a.prev_mutated_relative_spin      = self.a.mutated_relative_spin
                self.a.original_relative_velocity, self.a.original_relative_spin = action
                self.a.original_action = Action(velocity=self.a.original_relative_velocity, spin=self.a.original_relative_spin)
                self.a.absolute_action = Action(
                    velocity=clip(self.a.original_relative_velocity, min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity) * config.vehicle.controller_max_velocity,
                    spin=clip(self.a.original_relative_spin    , min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    ) * config.vehicle.controller_max_spin,
                )
                
                # 
                # logging and counter-increments
                # 
                if self.a.save_data and self.a.trajectory_file is not None:
                    self.a.trajectory_file.writelines(f"{self.a.spacial_info.x}, {self.a.spacial_info.y}, {self.a.spacial_info.angle}, {self.a.spacial_info.velocity}, {self.a.spacial_info.spin}, {self.a.original_relative_velocity}, {self.a.original_relative_spin}, {self.a.is_episode_start}\n")
                self.a.global_timestep += 1
                self.a.episode_steps = self.a.episode_steps + 1
                self.a.is_episode_start = 0
                
                # 
                # modify action
                # 
                if True:
                    # first force them to be within normal ranges
                    a.mutated_relative_velocity_action = clip(self.a.original_relative_velocity,  min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity)
                    a.mutated_relative_spin_action     = clip(self.a.original_relative_spin    ,  min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    )
                    
                    # 
                    # ADVERSITY
                    # 
                    if True:
                        # battery adversity
                        if config.simulator.battery_adversity_enabled:
                            self.a.simulated_battery_level *= 1-config.simulator.battery_decay_rate
                            self.a.recorder.add(timestep=self.a.global_timestep, simulated_battery_level=self.a.simulated_battery_level)
                            self.a.recorder.commit()
                            a.mutated_relative_velocity_action *= self.a.simulated_battery_level
                            # make sure velocity never goes negative (treat low battery as resistance)
                            a.mutated_relative_velocity_action = clip(a.mutated_relative_velocity_action,  min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity)
                        
                        # additive adversity
                        a.mutated_relative_velocity_action += config.simulator.velocity_offset
                        a.mutated_relative_spin_action     += config.simulator.spin_offset
                
                    # 
                    # add noise
                    # 
                    if config.simulator.use_gaussian_action_noise:
                        a.mutated_relative_velocity_action += random.normalvariate(mu=0, sigma=config.simulator.gaussian_action_noise.velocity_action.standard_deviation, )
                        a.mutated_relative_spin_action     += random.normalvariate(mu=0, sigma=config.simulator.gaussian_action_noise.spin_action.standard_deviation    , )
                    
                    # 
                    # action delay
                    # 
                    self.a.action_buffer.append((a.mutated_relative_velocity_action, a.mutated_relative_spin_action))
                    a.mutated_relative_velocity_action, a.mutated_relative_spin_action = self.a.action_buffer.pop(0) # ex: if 0 delay, this pop() will get what was just appended
                    
                    # 
                    # save
                    # 
                    self.a.mutated_relative_velocity = a.mutated_relative_velocity_action
                    self.a.mutated_relative_spin     = a.mutated_relative_spin_action
            # 
            # B
            # 
            if True:
                b = LazyDict()
                # 
                # push new action
                # 
                self.b.prev_original_relative_velocity = self.b.original_relative_velocity
                self.b.prev_original_relative_spin     = self.b.original_relative_spin
                self.b.prev_mutated_relative_velocity  = self.b.mutated_relative_velocity
                self.b.prev_mutated_relative_spin      = self.b.mutated_relative_spin
                self.b.original_relative_velocity, self.b.original_relative_spin = action
                self.b.absolute_action = Action(
                    clip(self.b.original_relative_velocity, min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity) * config.vehicle.controller_max_velocity,
                    clip(self.b.original_relative_spin    , min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    ) * config.vehicle.controller_max_spin
                )
                
                # 
                # logging and counter-increments
                # 
                if self.b.save_data and self.b.trajectory_file is not None:
                    self.b.trajectory_file.writelines(f"{self.b.spacial_info.x}, {self.b.spacial_info.y}, {self.b.spacial_info.angle}, {self.b.spacial_info.velocity}, {self.b.spacial_info.spin}, {self.b.original_relative_velocity}, {self.b.original_relative_spin}, {self.b.is_episode_start}\n")
                self.b.global_timestep += 1
                self.b.episode_steps = self.b.episode_steps + 1
                self.b.is_episode_start = 0
                
                # 
                # modify action
                # 
                if True:
                    # first force them to be within normal ranges
                    b.mutated_relative_velocity_action = clip(self.b.original_relative_velocity,  min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity)
                    b.mutated_relative_spin_action     = clip(self.b.original_relative_spin    ,  min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    )
                    
                    # 
                    # ADVERSITY
                    # 
                    if True:
                        # battery adversity
                        if config.simulator.battery_adversity_enabled:
                            self.b.simulated_battery_level *= 1-config.simulator.battery_decay_rate
                            self.b.recorder.add(timestep=self.b.global_timestep, simulated_battery_level=self.b.simulated_battery_level)
                            self.b.recorder.commit()
                            b.mutated_relative_velocity_action *= self.b.simulated_battery_level
                            # make sure velocity never goes negative (treat low battery as resistance)
                            b.mutated_relative_velocity_action = clip(b.mutated_relative_velocity_action,  min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity)
                        
                        # additive adversity
                        b.mutated_relative_velocity_action += config.simulator.velocity_offset
                        b.mutated_relative_spin_action     += config.simulator.spin_offset
                
                    # 
                    # add noise
                    # 
                    if config.simulator.use_gaussian_action_noise:
                        b.mutated_relative_velocity_action += random.normalvariate(mu=0, sigma=config.simulator.gaussian_action_noise.velocity_action.standard_deviation, )
                        b.mutated_relative_spin_action     += random.normalvariate(mu=0, sigma=config.simulator.gaussian_action_noise.spin_action.standard_deviation    , )
                    
                    # 
                    # action delay
                    # 
                    self.b.action_buffer.append((b.mutated_relative_velocity_action, b.mutated_relative_spin_action))
                    b.mutated_relative_velocity_action, b.mutated_relative_spin_action = self.b.action_buffer.pop(0) # ex: if 0 delay, this pop() will get what was just appended
                    
                    # 
                    # save
                    # 
                    self.b.mutated_relative_velocity = b.mutated_relative_velocity_action
                    self.b.mutated_relative_spin     = b.mutated_relative_spin_action
        
        # self.diff_compare(print_c=True, ignore=["recorder","action_duration","waypoints_list","waypoint_file_path","simulated_battery_level","trajectory_output_path","max_number_of_timesteps_per_episode"])
        
        # 
        # part2
        # 
        if True:
            # 
            # A
            # 
            if True:
                # 
                # modify spacial_info
                # 
                self.a.prev_spacial_info = self.a.spacial_info
                if type(override_next_spacial_info) != type(None):
                    # this is when the spacial_info is coming from the real world
                    self.a.spacial_info = override_next_spacial_info
                    # propogate changes to twist
                    self.a.twist.velocity = self.a.spacial_info.velocity
                    self.a.twist.spin = self.a.spacial_info.spin
                    self.a.pose.x = self.a.spacial_info.x
                    self.a.pose.y = self.a.spacial_info.y
                    self.a.pose.angle = self.a.spacial_info.angle
                else:
                    # initial
                    current_spacial_info = SpacialInformation(
                        x=self.a.pose.x,
                        y=self.a.pose.y,
                        angle=self.a.pose.angle,
                        velocity=self.a.twist.velocity,
                        spin=self.a.twist.spin,
                        timestep=self.a.episode_steps,
                    )
                
                    self.a.spacial_info = generate_next_spacial_info(
                        old_spacial_info=current_spacial_info,
                        relative_velocity=self.a.original_action.velocity,
                        relative_spin=self.a.original_action.spin,
                        action_duration=self.a.action_duration,
                        ep_poses=self.a.ep_poses,
                    ).with_timestep(self.a.episode_steps)
                    
                    self.a.twist = TwistEntry(
                        velocity=self.a.spacial_info.velocity,
                        spin=self.a.spacial_info.spin,
                        unknown=None,
                    )
                    self.a.pose = PoseEntry(
                        x=self.a.spacial_info.x,
                        y=self.a.spacial_info.y,
                        angle=self.a.spacial_info.angle,
                    )
                    self.a.prev_angle = self.a.spacial_info.angle
                    self.a.ep_poses.append(self.a.spacial_info)
                # 
                # increment waypoints
                # 
                a.closest_relative_index = 0
                a.last_waypoint_index = len(self.a.waypoints_list)-1
                a.next_waypoint          = self.a.waypoints_list[self.a.next_waypoint_index_new]
                if len(self.a.waypoints_list) > 1:
                    a.distance_to_waypoint       = get_distance(a.next_waypoint.x, a.next_waypoint.y, self.a.spacial_info.x, self.a.spacial_info.y)
                    a.got_further_away           = self.a.closest_distance < a.distance_to_waypoint
                    a.was_within_waypoint_radius = min(a.distance_to_waypoint, self.a.closest_distance) < config.simulator.waypoint_radius
                    # went past waypoint? increment the index
                    if a.distance_to_waypoint == 0 or (a.got_further_away and a.was_within_waypoint_radius):
                        a.closest_relative_index  = 1
                    # went past waypoint, but edgecase of getting further away:
                    elif a.was_within_waypoint_radius and self.a.next_waypoint_index_new < a.last_waypoint_index:
                        a.next_next_waypoint         = self.a.waypoints_list[self.a.next_waypoint_index_new+1]
                        a.waypoint_arm_angle = angle_created_by(
                            start=(self.a.spacial_info.x, self.a.spacial_info.y),
                            midpoint=(a.next_waypoint.x, a.next_waypoint.y),
                            end=(a.next_next_waypoint.x, a.next_next_waypoint.y),
                        )
                        a.we_passed_the_waypoint = a.waypoint_arm_angle < abs(math.degrees(90))
                        if a.we_passed_the_waypoint:
                            a.closest_relative_index  = 1
                            
                if a.closest_relative_index > 0:
                    self.a.next_waypoint_index_new += 1
                    # prevent indexing error
                    self.a.next_waypoint_index_new = min(self.a.next_waypoint_index_new, len(self.a.waypoints_list)-1)
                    a.next_waypoint = self.a.waypoints_list[self.a.next_waypoint_index_new]
                
                self.a.closest_distance = get_distance(
                    a.next_waypoint.x,
                    a.next_waypoint.y,
                    self.a.spacial_info.x,
                    self.a.spacial_info.y
                )
            # 
            # B
            # 
            if True:
                # 
                # modify spacial_info
                # 
                self.b.prev_spacial_info = self.b.spacial_info
                if type(override_next_spacial_info) != type(None):
                    # this is when the spacial_info is coming from the real world
                    self.b.spacial_info = override_next_spacial_info
                else:
                    # 
                    # apply action
                    # 
                    self.b.spacial_info = Helpers.generate_next_spacial_info(
                        old_spacial_info=SpacialInformation(*self.b.spacial_info),
                        relative_velocity=self.b.mutated_relative_velocity,
                        relative_spin=self.b.mutated_relative_spin,
                        action_duration=4*self.b.action_duration,
                    )
                
                # 
                # increment waypoints
                # 
                if True:
                    b.closest_relative_index = 0
                    b.last_waypoint_index = len(self.b.waypoints_list)-1
                    b.next_waypoint          = self.b.waypoints_list[self.b.next_waypoint_index]
                    if len(self.b.waypoints_list) > 1:
                        b.distance_to_waypoint       = get_distance(b.next_waypoint.x, b.next_waypoint.y, self.b.spacial_info.x, self.b.spacial_info.y)
                        b.got_further_away           = self.b.closest_distance < b.distance_to_waypoint
                        b.was_within_waypoint_radius = min(b.distance_to_waypoint, self.b.closest_distance) < config.simulator.waypoint_radius
                        # went past waypoint? increment the index
                        if b.distance_to_waypoint == 0 or (b.got_further_away and b.was_within_waypoint_radius):
                            b.closest_relative_index  = 1
                        # went past waypoint, but edgecase of getting further away:
                        elif b.was_within_waypoint_radius and self.b.next_waypoint_index < b.last_waypoint_index:
                            b.next_next_waypoint         = self.b.waypoints_list[self.b.next_waypoint_index+1]
                            b.waypoint_arm_angle = angle_created_by(
                                start=(self.b.spacial_info.x, self.b.spacial_info.y),
                                midpoint=(b.next_waypoint.x, b.next_waypoint.y),
                                end=(b.next_next_waypoint.x, b.next_next_waypoint.y),
                            )
                            b.we_passed_the_waypoint = b.waypoint_arm_angle < abs(math.degrees(90))
                            if b.we_passed_the_waypoint:
                                b.closest_relative_index  = 1
                                
                    if b.closest_relative_index > 0:
                        self.b.next_waypoint_index += 1
                        # prevent indexing error
                        self.b.next_waypoint_index = min(self.b.next_waypoint_index, len(self.b.waypoints_list)-1)
                        b.next_waypoint = self.b.waypoints_list[self.b.next_waypoint_index]
                    self.b.closest_distance = get_distance(
                        b.next_waypoint.x,
                        b.next_waypoint.y,
                        self.b.spacial_info.x,
                        self.b.spacial_info.y
                    )
            # 
            # C
            # 
            if True:
                # 
                # modify spacial_info
                # 
                self.c.prev_spacial_info = self.c.spacial_info
                if type(override_next_spacial_info) != type(None):
                    # this is when the spacial_info is coming from the real world
                    self.c.spacial_info = override_next_spacial_info
                else:
                    # 
                    # apply action
                    # 
                    self.c.spacial_info = Helpers.generate_next_spacial_info(
                        old_spacial_info=SpacialInformation(*self.c.spacial_info),
                        relative_velocity=self.c.mutated_relative_velocity,
                        relative_spin=self.c.mutated_relative_spin,
                        action_duration=4*self.c.action_duration,
                    )
                
                # 
                # increment waypoints
                # 
                change_in_waypoint_index, _ = advance_the_index_if_needed(
                    remaining_waypoints=self.c.waypoints_list[self.c.next_waypoint_index:],
                    x=self.c.spacial_info.x,
                    y=self.c.spacial_info.y,
                )
                self.c.next_waypoint_index += change_in_waypoint_index
                self.c.next_waypoint = c.next_waypoint = self.c.waypoints_list[self.c.next_waypoint_index]
                self.c.closest_distance = get_distance(
                    c.next_waypoint.x,
                    c.next_waypoint.y,
                    self.c.spacial_info.x,
                    self.c.spacial_info.y
                )
        
        self.diff_compare(print_c=True, ignore=["recorder","action_duration","waypoints_list","waypoint_file_path","simulated_battery_level","trajectory_output_path","max_number_of_timesteps_per_episode"])        
        exit()
        # 
        # part3
        # 
        if True:
            # 
            # A
            # 
            if True:
                omega_reward = (-2 * math.fabs(self.a.original_relative_spin))
                output, (
                    _,
                    self.a.crosstrack_error,
                    self.a.episode_steps,
                    omega_reward,
                    self.a.phi_error,
                    self.a.prev_absolute_action,
                    self.a.prev_next_waypoint_index,
                    self.a.reward,
                    self.a.total_episode_reward,
                    self.a.velocity_error,
                    self.a.velocity_reward,
                    _,
                    _,
                    _,
                    _,
                    self.a.next_waypoint_index,
                    self.a.ep_poses,
                ), other = pure_step(
                    relative_action=self.a.original_action,
                    absolute_action=self.a.absolute_action,
                    next_waypoint_index_=self.a.next_waypoint_index,
                    crosstrack_error=self.a.crosstrack_error,
                    action_duration=self.a.action_duration,
                    ep_poses=self.a.ep_poses,
                    episode_steps=self.a.episode_steps,
                    horizon=self.a.horizon,
                    max_number_of_timesteps_per_episode=self.a.max_number_of_timesteps_per_episode,
                    number_of_waypoints=self.a.number_of_waypoints,
                    omega_reward=omega_reward,
                    phi_error=self.a.phi_error,
                    pose=self.a.pose,
                    prev_absolute_action=self.a.prev_absolute_action,
                    prev_angle=self.a.prev_angle,
                    reward=self.a.reward,
                    total_episode_reward=self.a.total_episode_reward,
                    twist=self.a.twist,
                    velocity_error=self.a.velocity_error,
                    velocity_reward=self.a.velocity_reward,
                    waypoints_list=self.a.waypoints_list,
                )
                self.a.observation, self.a.reward, done, additional_info = output
                self.a.observation = Observation(self.a.observation+[self.a.global_timestep])
                self.a.renderer.render_if_needed(
                    prev_next_waypoint_index=self.a.prev_next_waypoint_index,
                    x_point=self.a.pose[0],
                    y_point=self.a.pose[1],
                    angle=self.a.pose[2],
                    text_data=f"""
                        velocity_error={self.a.velocity_error:.3f}
                        closest_index={self.a.next_waypoint_index_new}
                        crosstrack_error={self.a.crosstrack_error:.3f}
                        reward={self.a.reward:.4f}
                        warthog_vel={self.a.original_relative_velocity:.3f}
                        warthog_spin={self.a.original_relative_spin:.3f}
                        phi_error={self.a.phi_error*180/math.pi:.4f}
                        ep_reward={self.a.total_episode_reward:.4f}
                        
                        omega_reward={omega_reward:.4f}
                        velocity_reward={self.a.velocity_error:.4f}
                    """.replace("\n                ","\n"),
                )
            
            # 
            # B
            # 
            if False:
                # 
                # Reward Calculation
                # 
                self.b.reward, self.b.velocity_error, self.b.crosstrack_error, self.b.phi_error = Helpers.almost_original_reward_function(
                    spacial_info=self.b.spacial_info,
                    closest_distance=self.b.closest_distance,
                    relative_velocity=self.b.mutated_relative_velocity,
                    prev_relative_velocity=self.b.prev_mutated_relative_velocity,
                    relative_spin=self.b.mutated_relative_spin,
                    prev_relative_spin=self.b.prev_mutated_relative_spin,
                    closest_waypoint=b.next_waypoint,
                    closest_relative_index=b.closest_relative_index,
                )
                
                # 
                # make mutated observation
                # 
                if True:
                    self.b.prev_spacial_info_with_noise = self.b.spacial_info_with_noise
                    self.b.spacial_info_with_noise = self.b.spacial_info
                    # 
                    # add spacial noise
                    # 
                    if config.simulator.use_gaussian_spacial_noise:
                        self.b.spacial_info_with_noise = SpacialInformation(
                            self.b.spacial_info_with_noise.x        + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.x.standard_deviation       , ),
                            self.b.spacial_info_with_noise.y        + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.y.standard_deviation       , ),
                            self.b.spacial_info_with_noise.angle    + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.angle.standard_deviation   , ),
                            self.b.spacial_info_with_noise.velocity + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.spin.standard_deviation    , ),
                            self.b.spacial_info_with_noise.spin     + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.velocity.standard_deviation, ),
                            self.b.spacial_info_with_noise.timestep ,
                        )
                    
                    # generate observation off potentially incorrect (noisey) spacial info
                    prev_observation = self.b.observation
                    self.b.observation = Helpers.generate_observation(
                        closest_index=self.b.next_waypoint_index,
                        remaining_waypoints=self.b.waypoints_list[self.b.prev_next_waypoint_index:],
                        current_spacial_info=self.b.spacial_info_with_noise,
                    )
                
                # 
                # render
                # 
                # self.b.renderer.render_if_needed(
                #     prev_next_waypoint_index=self.b.prev_next_waypoint_index,
                #     x_point=self.b.spacial_info.x, # self.b.spacial_info.x
                #     y_point=self.b.spacial_info.y, # self.b.spacial_info.y
                #     angle=self.b.spacial_info.angle,   # self.b.spacial_info.angle
                #     text_data=f"vel_error={self.b.velocity_error:.3f}\nclosest_index={self.b.next_waypoint_index}\ncrosstrack_error={self.b.crosstrack_error:.3f}\nReward={self.b.reward:.4f}\nwarthog_vel={self.b.spacial_info.velocity:.3f}\nphi_error={self.b.phi_error*180/math.pi:.4f}\nep_reward={self.b.total_episode_reward:.4f}\n\nomega_reward={(-2 * math.fabs(self.b.original_relative_spin)):.4f}\nvel_reward={self.b.velocity_error:.4f}",
                # )
                
                additional_info = AdditionalInfo(
                    timestep_index=Unknown,
                    action_duration=self.b.action_duration,
                    spacial_info=self.b.prev_spacial_info,
                    spacial_info_with_noise=self.b.prev_spacial_info_with_noise,
                    observation_from_spacial_info_with_noise=prev_observation,
                    historic_transform=Unknown,
                    original_reaction=ReactionClass(self.b.original_relative_velocity, self.b.original_relative_spin ),
                    mutated_reaction=ReactionClass(self.b.mutated_relative_velocity, self.b.mutated_relative_spin ),
                    next_spacial_info=self.b.spacial_info,
                    next_spacial_info_spacial_info_with_noise=self.b.spacial_info_with_noise,
                    next_observation_from_spacial_info_with_noise=self.b.observation,
                    next_closest_index=self.b.next_waypoint_index,
                    reward=self.b.reward,
                )
                
                # 
                # done Calculation
                #
                done = False
                if self.b.next_waypoint_index >= len(self.b.waypoints_list) - 1:
                    done = True 
                if self.b.episode_steps == self.b.max_number_of_timesteps_per_episode:
                    done = True
                    self.b.episode_steps = 0
                # immediate end due to too much loss
                if config.simulator.allow_cut_short_episode:
                    if math.fabs(self.b.crosstrack_error) > magic_number_1_point_5 or math.fabs(self.b.phi_error) > magic_number_1_point_4:
                        done = True
                
                output = self.b.observation, self.b.reward, done, additional_info
            # 
            # C
            # 
            if True:
                # 
                # Reward Calculation
                # 
                self.c.reward, self.c.velocity_error, self.c.crosstrack_error, self.c.phi_error = Helpers.almost_original_reward_function(
                    spacial_info=self.c.spacial_info,
                    closest_distance=self.c.closest_distance,
                    relative_velocity=self.c.mutated_relative_velocity,
                    prev_relative_velocity=self.c.prev_mutated_relative_velocity,
                    relative_spin=self.c.mutated_relative_spin,
                    prev_relative_spin=self.c.prev_mutated_relative_spin,
                    closest_waypoint=b.next_waypoint,
                    closest_relative_index=b.closest_relative_index,
                )
                
                # 
                # make mutated observation
                # 
                if True:
                    self.c.prev_spacial_info_with_noise = self.c.spacial_info_with_noise
                    self.c.spacial_info_with_noise = self.c.spacial_info
                    # 
                    # add spacial noise
                    # 
                    if config.simulator.use_gaussian_spacial_noise:
                        self.c.spacial_info_with_noise = SpacialInformation(
                            self.c.spacial_info_with_noise.x        + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.x.standard_deviation       , ),
                            self.c.spacial_info_with_noise.y        + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.y.standard_deviation       , ),
                            self.c.spacial_info_with_noise.angle    + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.angle.standard_deviation   , ),
                            self.c.spacial_info_with_noise.velocity + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.spin.standard_deviation    , ),
                            self.c.spacial_info_with_noise.spin     + random.normalvariate(mu=0, sigma=config.simulator.gaussian_spacial_noise.velocity.standard_deviation, ),
                            self.c.spacial_info_with_noise.timestep ,
                        )
                    
                    # generate observation off potentially incorrect (noisey) spacial info
                    prev_observation = self.c.observation
                    print(f'''self.c.next_waypoint_index = {self.c.next_waypoint_index}''')
                    self.c.observation = Helpers.generate_observation(
                        closest_index=self.c.next_waypoint_index,
                        remaining_waypoints=self.c.waypoints_list[self.c.prev_next_waypoint_index:],
                        current_spacial_info=self.c.spacial_info_with_noise,
                    )
                
                # 
                # render
                # 
                # self.c.renderer.render_if_needed(
                #     prev_next_waypoint_index=self.c.prev_next_waypoint_index,
                #     x_point=self.c.spacial_info.x, # self.c.spacial_info.x
                #     y_point=self.c.spacial_info.y, # self.c.spacial_info.y
                #     angle=self.c.spacial_info.angle,   # self.c.spacial_info.angle
                #     text_data=f"vel_error={self.c.velocity_error:.3f}\nclosest_index={self.c.next_waypoint_index}\ncrosstrack_error={self.c.crosstrack_error:.3f}\nReward={self.c.reward:.4f}\nwarthog_vel={self.c.spacial_info.velocity:.3f}\nphi_error={self.c.phi_error*180/math.pi:.4f}\nep_reward={self.c.total_episode_reward:.4f}\n\nomega_reward={(-2 * math.fabs(self.c.original_relative_spin)):.4f}\nvel_reward={self.c.velocity_error:.4f}",
                # )
                
                additional_info = AdditionalInfo(
                    timestep_index=Unknown,
                    action_duration=self.c.action_duration,
                    spacial_info=self.c.prev_spacial_info,
                    spacial_info_with_noise=self.c.prev_spacial_info_with_noise,
                    observation_from_spacial_info_with_noise=prev_observation,
                    historic_transform=Unknown,
                    original_reaction=ReactionClass(self.c.original_relative_velocity, self.c.original_relative_spin ),
                    mutated_reaction=ReactionClass(self.c.mutated_relative_velocity, self.c.mutated_relative_spin ),
                    next_spacial_info=self.c.spacial_info,
                    next_spacial_info_spacial_info_with_noise=self.c.spacial_info_with_noise,
                    next_observation_from_spacial_info_with_noise=self.c.observation,
                    next_closest_index=self.c.next_waypoint_index,
                    reward=self.c.reward,
                )
                
                # 
                # done Calculation
                #
                done = False
                if self.c.next_waypoint_index >= len(self.c.waypoints_list) - 1:
                    done = True 
                if self.c.episode_steps == self.c.max_number_of_timesteps_per_episode:
                    done = True
                    self.c.episode_steps = 0
                # immediate end due to too much loss
                if config.simulator.allow_cut_short_episode:
                    if math.fabs(self.c.crosstrack_error) > magic_number_1_point_5 or math.fabs(self.c.phi_error) > magic_number_1_point_4:
                        done = True
                
                output = self.c.observation, self.c.reward, done, additional_info
        
        # self.diff_compare(print_c=True, ignore=["recorder","action_duration","waypoints_list","waypoint_file_path","simulated_battery_level","trajectory_output_path","max_number_of_timesteps_per_episode"])
        # exit()
        return output
    
    def reset(self, override_next_spacial_info=None):
        print(f'''resetting''')
        
        # 
        # stage 1
        # 
        if True:
            # 
            # A
            # 
            if True:
                self.a.is_episode_start = 1
                self.a.ep_poses = []
                self.a.total_episode_reward = 0
                
                index_a = config.simulator.starting_waypoint
                if config.simulator.starting_waypoint == 'random':
                    assert self.a.number_of_waypoints > 21
                    index_a = np.random.randint(self.a.number_of_waypoints-20, size=1)[0]
                    
                # if position is overriden by (most likely) the real world position
                if type(override_next_spacial_info) != type(None):
                    # this is when the spacial_info is coming from the real world
                    self.a.spacial_info = override_next_spacial_info
                    self.a.next_waypoint_index = index_a
                    self.a.prev_next_waypoint_index = index_a
                # simulator position
                else:
                    waypoint = self.a.waypoints_list[index_a]
                    self.a.next_waypoint_index = index_a
                    self.a.prev_next_waypoint_index = index_a
                    
                    self.a.spacial_info = SpacialInformation(
                        x=waypoint.x + config.simulator.random_start_position_offset,
                        y=waypoint.y + config.simulator.random_start_position_offset,
                        angle=waypoint.angle + config.simulator.random_start_angle_offset,
                        velocity=0,
                        spin=0,
                        timestep=0,
                    )
                    for desired_velocity, waypoint in zip(self.a.desired_velocities, self.a.waypoints_list):
                        waypoint.velocity = desired_velocity
            # 
            # B
            # 
            if True:
                self.b.is_episode_start = 1
                self.b.total_episode_reward = 0
                
                index_b = config.simulator.starting_waypoint
                if config.simulator.starting_waypoint == 'random':
                    assert len(self.b.waypoints_list) > 21
                    index_b = np.random.randint(len(self.b.waypoints_list)-20, size=1)[0]
                    
                # if position is overriden by (most likely) the real world position
                if type(override_next_spacial_info) != type(None):
                    # this is when the spacial_info is coming from the real world
                    self.b.spacial_info = override_next_spacial_info
                    self.b.next_waypoint_index = index_b
                    self.b.prev_next_waypoint_index = index_b
                # simulator position
                else:
                    waypoint = self.b.waypoints_list[index_b]
                    self.b.next_waypoint_index = index_b
                    self.b.prev_next_waypoint_index = index_b
                    
                    self.b.spacial_info = SpacialInformation(
                        x=waypoint.x + config.simulator.random_start_position_offset,
                        y=waypoint.y + config.simulator.random_start_position_offset,
                        angle=waypoint.angle + config.simulator.random_start_angle_offset,
                        velocity=0,
                        spin=0,
                        timestep=0,
                    )
                    for desired_velocity, waypoint in zip(self.b.desired_velocities, self.b.waypoints_list):
                        waypoint.velocity = desired_velocity
            # 
            # C
            # 
            if True:
                self.c.is_episode_start = 1
                self.c.ep_poses = []
                self.c.total_episode_reward = 0
                
                index_c = config.simulator.starting_waypoint
                if config.simulator.starting_waypoint == 'random':
                    size_limiter = 21
                    assert len(self.c.waypoints_list) > size_limiter
                    index_c = np.random.randint(len(self.c.waypoints_list)-(size_limiter-1), size=1)[0]
                
                # if position is overriden by (most likely) the real world position
                if type(override_next_spacial_info) != type(None):
                    # this is when the spacial_info is coming from the real world
                    self.c.spacial_info = override_next_spacial_info
                    self.c.next_waypoint_index = index_c
                    self.c.prev_next_waypoint_index = index_c
                # simulator position
                else:
                    waypoint = self.c.waypoints_list[index_c]
                    self.c.next_waypoint_index = index_c
                    self.c.prev_next_waypoint_index = index_c
                    
                    self.c.spacial_info = SpacialInformation(
                        x=waypoint.x + config.simulator.random_start_position_offset,
                        y=waypoint.y + config.simulator.random_start_position_offset,
                        angle=waypoint.angle + config.simulator.random_start_angle_offset,
                        velocity=0,
                        spin=0,
                        timestep=0,
                    )
                    for desired_velocity, waypoint in zip(self.c.desired_velocities, self.c.waypoints_list):
                        waypoint.velocity = desired_velocity
        
        
        # 
        # Stage 2
        #
        if True:
            # 
            # A
            # 
            if True:
                self.a.next_waypoint_index = index_a
                self.a.prev_next_waypoint_index = index_a
                self.a.pose = PoseEntry(
                    x=float(self.a.waypoints_list[index_a][0] + config.simulator.random_start_position_offset),
                    y=float(self.a.waypoints_list[index_a][1] + config.simulator.random_start_position_offset),
                    angle=float(self.a.waypoints_list[index_a][2] + config.simulator.random_start_angle_offset),
                )
                self.a.twist = TwistEntry(
                    velocity=0,
                    spin=0,
                    unknown=0,
                )
                # for i in range(0, self.a.number_of_waypoints):
                #     if self.a.desired_velocities[i] > self.a.max_vel:
                #         self.a.waypoints_list[i][3] = self.a.max_vel
                #     else:
                #         self.a.waypoints_list[i][3] = self.a.desired_velocities[i]
                # self.a.max_vel = 2
                # self.a.max_vel = self.a.max_vel + 1
                self.a.observation, self.a.closest_distance, self.a.next_waypoint_index = pure_get_observation(
                    next_waypoint_index_=self.a.next_waypoint_index,
                    horizon=self.a.horizon,
                    number_of_waypoints=self.a.number_of_waypoints,
                    pose=self.a.pose,
                    twist=self.a.twist,
                    waypoints_list=self.a.waypoints_list,
                )
                output = self.a.observation = Observation(self.a.observation+[self.c.spacial_info.timestep])
                self.a.renderer = Renderer(
                    vehicle_render_width=config.vehicle.render_width,
                    vehicle_render_length=config.vehicle.render_length,
                    waypoints_list=self.a.waypoints_list,
                    should_render=config.simulator.should_render and countdown(config.simulator.render_rate),
                    inital_x=self.a.pose[0],
                    inital_y=self.a.pose[1],
                    render_axis_size=20,
                    render_path=f"{config.output_folder}/render/",
                    history_size=config.simulator.number_of_trajectories,
                )
            # 
            # B
            # 
            if True:
                # 
                # calculate closest index
                # 
                closest_relative_index, self.b.closest_distance = Helpers.get_closest(
                    remaining_waypoints=self.b.waypoints_list[self.b.next_waypoint_index:],
                    x=self.b.spacial_info.x,
                    y=self.b.spacial_info.y,
                )
                self.b.next_waypoint_index += closest_relative_index
                self.b.prev_next_waypoint_index = self.b.next_waypoint_index
                
                output = self.b.observation = Helpers.generate_observation(
                    closest_index=self.b.next_waypoint_index,
                    remaining_waypoints=self.b.waypoints_list[self.b.next_waypoint_index:],
                    current_spacial_info=self.b.spacial_info,
                )
                
                # self.b.renderer = Renderer(
                #     vehicle_render_width=config.vehicle.render_width,
                #     vehicle_render_length=config.vehicle.render_length,
                #     waypoints_list=self.b.waypoints_list,
                #     should_render=config.simulator.should_render and countdown(config.simulator.render_rate),
                #     inital_x=self.b.spacial_info.x,
                #     inital_y=self.b.spacial_info.y,
                #     render_axis_size=20,
                #     render_path=f"{config.output_folder}/render/",
                #     history_size=config.simulator.number_of_trajectories,
                # )
                self.b.next_waypoint_index_ = self.b.next_waypoint_index # copy over
            
            # 
            # C
            # 
            if True:
                self.c.next_waypoint_index = index_c
                self.c.prev_next_waypoint_index = index_c
                self.c.pose = PoseEntry(
                    x=float(self.c.waypoints_list[index_c][0] + config.simulator.random_start_position_offset),
                    y=float(self.c.waypoints_list[index_c][1] + config.simulator.random_start_position_offset),
                    angle=float(self.c.waypoints_list[index_c][2] + config.simulator.random_start_angle_offset),
                )
                self.c.twist = TwistEntry(
                    velocity=0,
                    spin=0,
                    unknown=0,
                )
                
                # 
                # calculate closest index
                # 
                change_in_waypoint_index_c, closest_distance_c = advance_the_index_if_needed(
                    remaining_waypoints=self.c.waypoints_list[self.c.next_waypoint_index:],
                    x=self.c.pose.x,
                    y=self.c.pose.y,
                )
                # self.c.next_waypoint_index += change_in_waypoint_index
                
                closest_relative_index, self.c.closest_distance = Helpers.get_closest(
                    remaining_waypoints=self.c.waypoints_list[self.c.next_waypoint_index:],
                    x=self.c.spacial_info.x,
                    y=self.c.spacial_info.y,
                )
                
                self.c.next_waypoint_index += closest_relative_index
                self.c.prev_next_waypoint_index = self.c.next_waypoint_index
                
                obs_c = SimpleHelpers.generate_observation_array(
                    self.c.next_waypoint_index,
                    config.simulator.horizon,
                    len(self.c.waypoints_list),
                    self.c.pose,
                    self.c.twist,
                    self.c.waypoints_list,
                )
                
                output = self.c.prev_observation = Helpers.generate_observation(
                    closest_index=self.c.next_waypoint_index,
                    remaining_waypoints=self.c.waypoints_list[self.c.next_waypoint_index:],
                    current_spacial_info=self.c.spacial_info,
                )
                
            pass
        
        return output
    
    def diff_compare(self, print_c=False, ignore=[], against_c=False):
        print(f'''diff_compare:''')
        shared_keys = list(set(self.b.keys()) & set(self.a.keys()))
        only_in_a = list(set(self.a.keys()) - set(self.b.keys()))
        only_in_b = list(set(self.b.keys()) - set(self.a.keys()))
        equal_keys = []
        b = self.c if against_c else self.b
        print("    differences:")
        helper_for_ordering = [ each for each in [
            "prev_original_relative_velocity",
            "prev_original_relative_spin",
            "prev_mutated_relative_velocity",
            "prev_mutated_relative_spin",
            "original_relative_velocity",
            "original_relative_spin",
            "absolute_action",
            "episode_steps",
            "is_episode_start",
            "mutated_relative_velocity",
            "mutated_relative_spin",
            "prev_spacial_info",
            "spacial_info",
            "next_waypoint_index",
            "closest_distance",
            "reward",
            "velocity_error",
            "crosstrack_error",
            "phi_error",
            "prev_spacial_info_with_noise",
            "spacial_info_with_noise",
            "observation",
        ] if each in shared_keys ]
        shared_keys = no_duplicates(helper_for_ordering + shared_keys)
        for each in shared_keys:
            if each in ignore:
                continue
            if not are_equal(self.a[each], b[each]):
                print(f'''        {each}:\n            {json.dumps(repr(self.a[each]))[1:-1]}\n            {json.dumps(repr(b[each]))[1:-1]}''')
                if type(self.a[each]) != type(b[each]):
                    print(f"            NOTE: types are not equal: {type(self.a[each])}!={type(b[each])}")
            else:
                equal_keys.append(each)
        
        if print_c:
            print("    self.c.update({")
            for each in equal_keys:
                print(f'''        {repr(each)}: {repr(self.a[each])},''')
            print("    })")
            
            print("    match-up helper:")
            for each in sorted([f"{each}:a" for each in only_in_a] + [f"{each}:b" for each in only_in_b]):
                print(f'''        {each}''')


if not grug_test.fully_disable and (grug_test.replay_inputs or grug_test.record_io):
    @grug_test(skip=False)
    def smoke_test_warthog(trajectory_file):
        actual_starting_setting = config.simulator.starting_waypoint
        config.simulator.starting_waypoint = 0 # force override it for test
        try:
            config.simulator.should_render = False
            env = WarthogEnv(path_to.waypoints_folder+f"/{trajectory_file}")
            env.should_render = False
            outputs = []
            def env_snapshot(env):
                return deepcopy(dict(
                        waypoints_list=                                getattr(env._, "waypoints_list"                                , None),
                        spacial_info=                                  getattr(env._, "spacial_info"                                  , None),
                        pose=                                          getattr(env._, "pose"                                          , None),
                        twist=                                         getattr(env._, "twist"                                         , None),
                        next_waypoint_index=                           getattr(env._, "next_waypoint_index"                           , None),
                        next_waypoint_index_=                                 getattr(env._, "next_waypoint_index_"                                 , None),
                        prev_next_waypoint_index_=                            getattr(env._, "prev_next_waypoint_index_"                            , None),
                        closest_distance=                              getattr(env._, "closest_distance"                              , None),
                        number_of_waypoints=                           getattr(env._, "number_of_waypoints"                           , None),
                        horizon=                                       getattr(env._, "horizon"                                       , None),
                        action_duration=                               getattr(env._, "action_duration"                               , None),
                        desired_velocities=                            getattr(env._, "desired_velocities"                            , None),
                        max_vel=                                       getattr(env._, "max_vel"                                       , None),
                        warthog_length=                                getattr(env._, "warthog_length"                                , None),
                        warthog_width=                                 getattr(env._, "warthog_width"                                 , None),
                        diag_angle=                                    getattr(env._, "diag_angle"                                    , None),
                        prev_angle=                                    getattr(env._, "prev_angle"                                    , None),
                        crosstrack_error=                              getattr(env._, "crosstrack_error"                              , None),
                        velocity_error=                                     getattr(env._, "velocity_error"                                     , None),
                        phi_error=                                     getattr(env._, "phi_error"                                     , None),
                        episode_steps=                                 getattr(env._, "episode_steps"                                 , None),
                        max_number_of_timesteps_per_episode=           getattr(env._, "max_number_of_timesteps_per_episode"           , None),
                        total_episode_reward=                               getattr(env._, "total_episode_reward"                               , None),
                        reward=                                        getattr(env._, "reward"                                        , None),
                        original_relative_velocity=                    getattr(env._, "original_relative_velocity"                    , None),
                        original_relative_spin=                        getattr(env._, "original_relative_spin"                        , None),
                        action=                                        getattr(env._, "action"                                        , None),
                        absolute_action=                               getattr(env._, "absolute_action"                               , None),
                        reaction=                                      getattr(env._, "reaction"                                      , None),
                        prev_absolute_action=                          getattr(env._, "prev_absolute_action"                          , None),
                        action_buffer=                                 getattr(env._, "action_buffer"                                 , None),
                        prev_reaction=                                 getattr(env._, "prev_reaction"                                 , None),
                        velocity_reward=                                    getattr(env._, "velocity_reward"                                    , None),
                        is_delayed_dynamics=                           getattr(env._, "is_delayed_dynamics"                           , None),
                        v_delay_data=                                  getattr(env._, "v_delay_data"                                  , None),
                        w_delay_data=                                  getattr(env._, "w_delay_data"                                  , None),
                        save_data=                                     getattr(env._, "save_data"                                     , None),
                        is_episode_start=                              getattr(env._, "is_episode_start"                              , None),
                        ep_poses=                                      getattr(env._, "ep_poses"                                      , None),
                    ))
            
            outputs.append(env_snapshot(env))
            env.step([0.5, 0.5])
            outputs.append(env_snapshot(env))
            env.step([0.56, 0.56])
            outputs.append(env_snapshot(env))
            env.step([0.567, 0.567])
            outputs.append(env_snapshot(env))
            env.step([0.5678, 0.5678])
            outputs.append(env_snapshot(env))
            return outputs
        finally:
            config.simulator.starting_waypoint = actual_starting_setting
    
    if grug_test.replay_inputs:
        smoke_test_warthog("real1.csv")
    # exit()
global_a_observation_buffer = []
global_b_observation_buffer = []
class SimpleHelpers:
    @staticmethod
    @grug_test(max_io=5, skip=False)
    def generate_observation_array(
        next_waypoint_index_,
        horizon,
        number_of_waypoints,
        pose,
        twist,
        waypoints_list,
        **kwargs,
    ):
        observation   = [0] * (horizon * 4 + 2)
        index = 0
        global_a_observation_buffer.clear()
        for horizon_index in range(0, horizon):
            waypoint_index = horizon_index + next_waypoint_index_
            if waypoint_index < number_of_waypoints:
                waypoint = waypoints_list[waypoint_index]
                gap_of_distance = get_distance(x1=waypoint.x, y1=waypoint.y, x2=pose.x, y2=pose.y)
                x_diff = waypoint.x - pose.x
                y_diff = waypoint.y - pose.y
                angle_to_next_point = get_angle_from_origin(x_diff, y_diff)
                current_angle = zero_to_2pi(pose.angle)
                gap_of_desired_angle_at_next = pi_to_pi(waypoint.angle - current_angle)
                gap_of_angle_directly_towards_next = pi_to_pi(angle_to_next_point - current_angle)
                gap_of_velocity = waypoint.velocity - twist.velocity
                global_a_observation_buffer.append(LazyDict(
                    waypoint=waypoint,
                    gap_of_distance=gap_of_distance,
                    x_diff=x_diff,
                    y_diff=y_diff,
                    angle_to_next_point=angle_to_next_point,
                    current_angle=current_angle,
                    pose=pose,
                    gap_of_desired_angle_at_next=gap_of_desired_angle_at_next,
                ))
                observation[index]     = gap_of_distance
                observation[index + 1] = gap_of_angle_directly_towards_next
                observation[index + 2] = gap_of_desired_angle_at_next
                observation[index + 3] = gap_of_velocity
            else:
                observation[index] = 0.0
                observation[index + 1] = 0.0
                observation[index + 2] = 0.0
                observation[index + 3] = 0.0
            index = index + 4
        observation[index] = twist.velocity
        observation[index + 1] = twist.spin
        
        return observation

class Helpers:
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
        global_b_observation_buffer.clear()
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
            
            global_b_observation_buffer.append(LazyDict(
                waypoint=waypoint,
                gap_of_distance=gap_of_distance,
                x_diff=x_diff,
                y_diff=y_diff,
                angle_to_next_point=angle_to_next_point,
                current_angle=current_angle,
                pose=current_spacial_info,
                gap_of_desired_angle_at_next=gap_of_desired_angle_at_next,
            ))
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
            global_b_next_spacial_buffer.append(next_spacial_info)
            
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
                if closest_waypoint.velocity >= min_waypoint_speed: # old code: self.b.waypoints_list[k][3] >= 2.5
                    # if rule is broken, no reward
                    if abs_velocity_error > max_error_allowed: # old code: math.fabs(self.vel_error) > 1.5
                        running_reward = 0
                        break
        
        return running_reward, velocity_error, crosstrack_error, phi_error
    
    @staticmethod
    @grug_test(max_io=30, record_io=None, additional_io_per_run=None, skip=True)
    def almost_original_reward_function(**kwargs):
        closest_relative_index = kwargs["closest_relative_index"]
        running_reward, *other = Helpers.original_reward_function(**kwargs)
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