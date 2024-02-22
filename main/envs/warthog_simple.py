import math
import time
import csv
from copy import deepcopy
from collections import namedtuple
import json

from gym import spaces
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import gym
import matplotlib as mpl
import numpy as np
from config import grug_test, path_to, config
from envs.warthog import read_waypoint_file, SpacialInformation


from generic_tools.plotting import create_slider_from_traces
from generic_tools.geometry import zero_to_2pi, pi_to_pi
from __dependencies__.blissful_basics import Csv, create_named_list_class, FS, print, stringify, clip, countdown, LazyDict
from __dependencies__.grug_test import register_named_tuple
from misc import are_equal

from config import config, path_to, grug_test
from generic_tools.geometry import get_distance, get_angle_from_origin, zero_to_2pi, pi_to_pi, abs_angle_difference, angle_created_by

from data_structures import Unknown, Action, StepOutput, StepSideEffects, GetObservationOutput, RewardOutput, SimWarthogOutput, PoseEntry, TwistEntry, SpacialHistory, SpacialInformation, ReactionClass, WaypointGap, Waypoint, Observation, AdditionalInfo


@grug_test(max_io=5, skip=False)
def generate_next_spacial_info(
    old_spacial_info,
    relative_velocity,
    relative_spin,
    action_duration,
    ep_poses,
    **kwargs,
):
    absolute_action = Action(
        np.clip(relative_velocity, 0, 1) * 4.0,
        np.clip(relative_spin, -1, 1) * 2.5,
    )
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
    
    for each in range(config.simulator.granularity_of_calculations):
        old_x, old_y, prev_angle = pose
        old_v, old_w, *unknown = twist
        twist = TwistEntry(
            velocity=float(absolute_velocity),
            spin=float(absolute_spin),
            unknown=(None if len(unknown) == 0 else unknown[0]),
        )
        pose = PoseEntry(
            x=float(old_x + old_v * math.cos(prev_angle) * action_duration),
            y=float(old_y + old_v * math.sin(prev_angle) * action_duration),
            angle=float(zero_to_2pi(prev_angle + old_w * action_duration)),
        )
    
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

@grug_test(max_io=5, skip=False)
def pure_get_observation(
    next_waypoint_index,
    horizon,
    number_of_waypoints,
    pose,
    twist,
    waypoints_list,
    **kwargs,
):
    obs   = [0] * (horizon * 4 + 2)
    index   = next_waypoint_index
    
    closest_distance = math.inf
    for i in range(next_waypoint_index, number_of_waypoints):
        dist = get_distance(x1=waypoints_list[i][0], y1=waypoints_list[i][1], x2=pose[0], y2=pose[1])
        if dist <= closest_distance:
            closest_distance = dist
            index = i
        else:
            break
    next_waypoint_index = index
    
    j = 0
    for i in range(0, horizon):
        k = i + next_waypoint_index
        if k < number_of_waypoints:
            r = get_distance(x1=waypoints_list[k][0], y1=waypoints_list[k][1], x2=pose[0], y2=pose[1])
            xdiff = waypoints_list[k][0] - pose[0]
            ydiff = waypoints_list[k][1] - pose[1]
            th = get_angle_from_origin(xdiff, ydiff)
            vehicle_th = zero_to_2pi(pose[2])
            yaw_error = pi_to_pi(waypoints_list[k][2] - vehicle_th)
            vel = waypoints_list[k][3]
            obs[j] = r
            obs[j + 1] = pi_to_pi(th - vehicle_th)
            obs[j + 2] = yaw_error
            obs[j + 3] = vel - twist[0]
        else:
            obs[j] = 0.0
            obs[j + 1] = 0.0
            obs[j + 2] = 0.0
            obs[j + 3] = 0.0
        j = j + 4
    obs[j] = twist[0]
    obs[j + 1] = twist[1]
    
    return GetObservationOutput(obs, closest_distance, next_waypoint_index)

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
    total_ep_reward,
    next_waypoint_index,
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
    xdiff         = waypoints_list[next_waypoint_index][0] - pose[0]
    ydiff         = waypoints_list[next_waypoint_index][1] - pose[1]
    yaw_error     = pi_to_pi(get_angle_from_origin(xdiff, ydiff) - pose[2])
    omega_reward  = -2 * math.fabs(action[1])
    velocity_reward    = -math.fabs(action[0] - prev_absolute_action[0])
    
    reward, velocity_error, crosstrack_error, phi_error = pure_reward(
        closest_waypoint=waypoints_list[next_waypoint_index],
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
    
    total_ep_reward = total_ep_reward + reward
    prev_absolute_action = action
    
    return reward, crosstrack_error, xdiff, ydiff, yaw_error, phi_error, velocity_error, done, episode_steps, omega_reward, velocity_reward, prev_absolute_action, total_ep_reward


@grug_test(max_io=10, skip=False)
def pure_step(
    relative_action,
    absolute_action,
    next_waypoint_index,
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
    total_ep_reward,
    twist,
    velocity_error,
    velocity_reward,
    waypoints_list,
    **kwargs,
):
    prev_next_waypoint_index = next_waypoint_index
    obs, closest_distance, next_waypoint_index = pure_get_observation(
        next_waypoint_index=next_waypoint_index,
        horizon=horizon,
        number_of_waypoints=number_of_waypoints,
        pose=pose,
        twist=twist,
        waypoints_list=waypoints_list,
    )
    done = False
    if next_waypoint_index >= number_of_waypoints - 1:
        done = True
    
    reward, crosstrack_error, xdiff, ydiff, yaw_error, phi_error, velocity_error, done, episode_steps, omega_reward, velocity_reward, prev_absolute_action, total_ep_reward = pure_reward_wrapper(
        total_ep_reward=total_ep_reward,
        next_waypoint_index=next_waypoint_index,
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
            prev_next_waypoint_index,
            reward,
            total_ep_reward,
            velocity_error,
            velocity_reward,
            twist,
            prev_angle,
            pose,
            closest_distance,
            next_waypoint_index,
            ep_poses,
        ),
        (
            yaw_error,
            ydiff,
            xdiff,
        )
    )

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
    
    render_axis_size = 20
    
    def __init__(self, waypoint_file_path, trajectory_output_path=None, recorder=None, *args, **kwargs):
        super(WarthogEnv, self).__init__()
        self.a = LazyDict()
        self.b = LazyDict()
        self.c = LazyDict()
        self.log = LazyDict()
        # 
        # B
        # 
        if True:
            self.b.waypoint_file_path = waypoint_file_path
            self.b.out_trajectory_file = trajectory_output_path
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
            self.b.number_of_trajectories = config.simulator.number_of_trajectories
            self.b.render_axis_size       = 20
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
            self.b.prev_observation        = None
            self.b.is_episode_start        = 1
            self.b.out_trajectory_file         = None
            self.b.global_timestep         = 0
            self.b.action_buffer           = [ (0,0) ] * config.simulator.action_delay # seed the buffer with delays
            self.b.simulated_battery_level = 1.0 # proportion 
            
            if self.b.waypoint_file_path is not None:
                self.b.desired_velocities, self.b.waypoints_list = read_waypoint_file(self.b.waypoint_file_path)
            
            self.b.x_pose = [0.0] * self.b.number_of_trajectories
            self.b.y_pose = [0.0] * self.b.number_of_trajectories
            self.b.crosstrack_error = 0
            self.b.velocity_error   = 0
            self.b.phi_error        = 0
            self.b.prev_render_timestamp   = time.time()
            self.b.should_render    = config.simulator.should_render and countdown(config.simulator.render_rate)
        
        # 
        # A
        # 
        if True:
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
            self.a.waypoints_dist = 0.5
            self.a.warthog_length = 0.5 / 2.0
            self.a.warthog_width = 1.0 / 2.0
            self.a.warthog_diag = math.sqrt(
                self.a.warthog_width**2 + self.a.warthog_length**2
            )
            self.a.diag_angle              = math.atan2(self.a.warthog_length, self.a.warthog_width)
            self.a.prev_angle              = 0
            self.a.number_of_trajectories                  = 100
            self.a.x_pose                   = [0.0] * self.a.number_of_trajectories
            self.a.y_pose                   = [0.0] * self.a.number_of_trajectories
            self.a.crosstrack_error        = 0
            self.a.velocity_error               = 0
            self.a.phi_error               = 0
            self.a.start_step_for_sup_data = 500000
            self.a.max_number_of_timesteps_per_episode            = 700
            self.a.total_episode_reward         = 0
            self.a.reward                  = 0
            self.a.action                  = [0.0, 0.0]
            self.a.absolute_action         = [0.0, 0.0]
            self.a.prev_absolute_action    = [0.0, 0.0]
            self.a.omega_reward            = 0
            self.a.velocity_reward              = 0
            self.a.delay_steps             = 5
            self.a.save_data               = False
            self.a.ep_dist                 = 0
            self.a.ep_poses                = []
            
            self.a.should_render = True
            self.a.global_timestep = 0
            
            if self.a.should_render:
                from matplotlib.patches import Rectangle
                self.a.warthog_diag   = math.sqrt(config.vehicle.render_width**2 + config.vehicle.render_length**2)
                self.a.diagonal_angle = math.atan2(config.vehicle.render_length, config.vehicle.render_width)
                self.a.prev_render_timestamp = time.time()
                
                self.a.render_path = f"{config.output_folder}/render/"
                print(f'''rendering to: {self.a.render_path}''')
                FS.remove(self.a.render_path)
                FS.ensure_is_folder(self.a.render_path)
                plt.ion
                self.a.fig = plt.figure(dpi=100, figsize=(10, 10))
                self.a.ax  = self.a.fig.add_subplot(111)
                self.a.ax.set_xlim([-4, 4])
                self.a.ax.set_ylim([-4, 4])
                self.a.rect = Rectangle((0.0, 0.0), config.vehicle.render_width * 2, config.vehicle.render_length * 2, fill=False)
                self.a.ax.add_artist(self.a.rect)
                (self.a.cur_pos,) = self.a.ax.plot(self.a.x_pose, self.a.y_pose, "+g")
                self.a.text = self.a.ax.text(1, 2, f"velocity_error={self.a.velocity_error}", style="italic", bbox={"facecolor": "red", "alpha": 0.5, "pad": 10}, fontsize=12)
                x = []
                y = []
                for each_x, each_y, *_ in self.a.waypoints_list:
                    x.append(each_x)
                    y.append(each_y)
                self.a.ax.plot(x, y, "+b")
            
        #
        # C
        #
        if True:
            self.c.save_data                           = config.simulator.save_data
            self.c.max_number_of_timesteps_per_episode = config.simulator.max_number_of_timesteps_per_episode
            self.c.number_of_trajectories              = config.simulator.number_of_trajectories
            self.c.action_duration                     = config.simulator.action_duration
            
            self.c.waypoint_file_path                  = waypoint_file_path
            
            self.c.prev_render_timestamp                      = time.time()
            self.c.should_render                       = config.simulator.should_render and countdown(config.simulator.render_rate)
            self.c.desired_velocities, self.c.waypoints_list = read_waypoint_file(waypoint_file_path)
            self.c.number_of_waypoints = len(self.c.waypoints_list)
            self.c.next_waypoint_index      = 0
            self.c.global_timestep          = 0
            self.c.prev_next_waypoint_index = 0
            self.c.phi_error                = 0
            self.c.crosstrack_error         = 0
            self.c.velocity_error           = 0
            self.c.reward                   = 0
            self.c.total_episode_reward     = 0
            self.c.closest_distance         = math.inf
            self.c.x_pose                   = [0.0] * self.c.number_of_trajectories
            self.c.y_pose                   = [0.0] * self.c.number_of_trajectories
            
        self.diff_compare(
            print_c=True,
            ignore=[
                "save_data",
                "max_number_of_timesteps_per_episode",
                "number_of_trajectories",
                "action_duration",
                "waypoint_file_path",
                "prev_render_timestamp",
                "should_render",
                "desired_velocities",
                "waypoints_list",
                "number_of_waypoints",
                "global_timestep",
                "prev_next_waypoint_index",
                "phi_error",
                "crosstrack_error",
                "velocity_error",
                "reward",
                "total_episode_reward",
                "closest_distance",
                "x_pose",
                "y_pose",
                "next_waypoint_index",
            ]
        )
        exit()
        self.reset()
    
    def __del__(self):
        if self.b.out_trajectory_file:
            self.b.out_trajectory_file.close()
    
    def diff_compare(self, print_c=False, ignore=[]):
        print(f'''diff_compare:''')
        shared_keys = list(set(self.b.keys()) & set(self.a.keys()))
        only_in_a = list(set(self.a.keys()) - set(self.b.keys()))
        only_in_b = list(set(self.b.keys()) - set(self.a.keys()))
        equal_keys = []
        print("    differences:")
        for each in shared_keys:
            if each in ignore:
                continue
            if not are_equal(self.a[each], self.b[each]):
                print(f'''        {each}:\n            {json.dumps(repr(self.a[each]))[1:-1]}\n            {json.dumps(repr(self.b[each]))[1:-1]}''')
                if type(self.a[each]) != type(self.b[each]):
                    print(f"            NOTE: types are not equal: {type(self.a[each])}!={type(self.b[each])}")
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
    
    @property
    def _(self):
        return LazyDict({
            key: getattr(self.a, key, getattr(self.b, key, None)) for key in [
                "absolute_action",
                "action",
                "action_buffer",
                "action_duration",
                "closest_distance",
                "crosstrack_error",
                "cur_pos",
                "delay_steps",
                "desired_velocities",
                "diag_angle",
                "diagonal_angle",
                "ep_dist",
                "ep_poses",
                "episode_steps",
                "global_timestep",
                "horizon",
                "is_episode_start",
                "max_number_of_timesteps_per_episode",
                "max_vel",
                "mutated_relative_spin",
                "mutated_relative_velocity",
                "next_waypoint_index",
                "next_waypoint_index",
                "number_of_trajectories",
                "number_of_waypoints",
                "omega_reward",
                "original_relative_spin",
                "original_relative_velocity",
                "phi_error",
                "pose",
                "prev_absolute_action",
                "prev_angle",
                "prev_mutated_relative_spin",
                "prev_mutated_relative_velocity",
                "prev_next_waypoint_index",
                "prev_original_relative_spin",
                "prev_original_relative_velocity",
                "prev_spacial_info",
                "prev_render_timestamp",
                "random_start_angle_offset",
                "random_start_position_offset",
                "relative_action",
                "render_axis_size",
                "render_path",
                "reward",
                "save_data",
                "should_render",
                "spacial_info",
                "start_step_for_sup_data",
                "total_episode_reward",
                "twist",
                "velocity_error",
                "velocity_reward",
                "warthog_diag",
                "warthog_length",
                "warthog_width",
                "waypoints_dist",
                "waypoints_list",
                "x_pose",
                "y_pose",
                "observation",
                "prev_observation",
                "prev_spacial_info_with_noise",
                "recorder",
                "simulated_battery_level",
                "spacial_info_with_noise",
                "velocity_error",
                "waypoint_file_path",
            ]
        })
    
    # just a wrapper around the pure_step
    def step(self, action, override_next_spacial_info=None):
        #  
        # push new action
        # 
        self.b.prev_original_relative_velocity = self.b.original_relative_velocity
        self.b.prev_original_relative_spin     = self.b.original_relative_spin
        self.b.prev_mutated_relative_velocity  = self.b.mutated_relative_velocity
        self.b.prev_mutated_relative_spin      = self.b.mutated_relative_spin
        self.b.original_relative_velocity, self.b.original_relative_spin = action
        self.a.absolute_action = Action(
            velocity=clip(self.b.original_relative_velocity, min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity) * config.vehicle.controller_max_velocity,
            spin=clip(self.b.original_relative_spin    , min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    ) * config.vehicle.controller_max_spin,
        )
        
        # 
        # logging and counter-increments
        # 
        if self.a.save_data and self.a.out_trajectory_file is not None:
            self.a.out_trajectory_file.writelines(f"{self.a.spacial_info.x}, {self.a.spacial_info.y}, {self.a.spacial_info.angle}, {self.a.spacial_info.velocity}, {self.a.spacial_info.spin}, {self.b.original_relative_velocity}, {self.b.original_relative_spin}, {self.a.is_episode_start}\n")
        self.a.global_timestep += 1
        self.a.episode_steps = self.a.episode_steps + 1
        self.a.is_episode_start = 0
        
        # 
        # modify action
        # 
        if True:
            # first force them to be within normal ranges
            mutated_relative_velocity_action = clip(self.b.original_relative_velocity,  min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity)
            mutated_relative_spin_action     = clip(self.b.original_relative_spin    ,  min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    )
            
            # 
            # ADVERSITY
            # 
            if True:
                # battery adversity
                if config.simulator.battery_adversity_enabled:
                    self.b.simulated_battery_level *= 1-config.simulator.battery_decay_rate
                    self.b.recorder.add(timestep=self.a.global_timestep, simulated_battery_level=self.b.simulated_battery_level)
                    self.b.recorder.commit()
                    mutated_relative_velocity_action *= self.b.simulated_battery_level
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
            self.b.action_buffer.append((mutated_relative_velocity_action, mutated_relative_spin_action))
            mutated_relative_velocity_action, mutated_relative_spin_action = self.b.action_buffer.pop(0) # ex: if 0 delay, this pop() will get what was just appended
            
            # 
            # save
            # 
            self.b.mutated_relative_velocity = mutated_relative_velocity_action
            self.b.mutated_relative_spin     = mutated_relative_spin_action
        
        
        # 
        # modify spacial_info
        # 
        self.a.prev_spacial_info = self.a.spacial_info
        if type(override_next_spacial_info) != type(None):
            # this is when the spacial_info is coming from the real world
            self.a.spacial_info = override_next_spacial_info
        else:
            # FIXME:
            # self.a.spacial_info = generate_next_spacial_info(
            #     old_spacial_info=SpacialInformation(*self.a.spacial_info),
            #     relative_velocity=self.b.mutated_relative_velocity,
            #     relative_spin=self.b.mutated_relative_spin,
            #     action_duration=self.a.action_duration,
            # )
            pass
            
        self.a.global_timestep += 1
        self.a.action = action
        self.a.relative_action = Action(velocity=self.b.original_relative_velocity, spin=self.b.original_relative_spin) 
        
        self.a.spacial_info = SpacialInformation(
            x=self.a.pose.x,
            y=self.a.pose.y,
            angle=self.a.pose.angle,
            velocity=self.a.twist.velocity,
            spin=self.a.twist.spin,
            timestep=self.a.episode_steps,
        )
        
        self.a.twist, self.a.prev_angle, self.a.pose, ep_poses, _ = generate_next_spacial_info(
            old_spacial_info=self.a.spacial_info,
            relative_velocity=self.a.relative_action.velocity,
            relative_spin=self.a.relative_action.spin,
            action_duration=self.a.action_duration,
            ep_poses=self.a.ep_poses,
        )
        
        self.a.spacial_info = SpacialInformation(
            x=self.a.pose.x,
            y=self.a.pose.y,
            angle=self.a.pose.angle,
            velocity=self.a.twist.velocity,
            spin=self.a.twist.spin,
            timestep=self.a.episode_steps,
        )
        
        closest_relative_index = 0
        last_waypoint_index = len(self.a.waypoints_list)-1
        next_waypoint          = self.a.waypoints_list[self.a.next_waypoint_index]
        if len(self.a.waypoints_list) > 1:
            distance_to_waypoint       = get_distance(next_waypoint.x, next_waypoint.y, self.a.spacial_info.x, self.a.spacial_info.y)
            got_further_away           = self.a.closest_distance < distance_to_waypoint
            was_within_waypoint_radius = min(distance_to_waypoint, self.a.closest_distance) < config.simulator.waypoint_radius
            # went past waypoint? increment the index
            if distance_to_waypoint == 0 or (got_further_away and was_within_waypoint_radius):
                closest_relative_index  = 1
            # went past waypoint, but edgecase of getting further away:
            elif was_within_waypoint_radius and self.a.next_waypoint_index < last_waypoint_index:
                next_next_waypoint         = self.a.waypoints_list[self.a.next_waypoint_index+1]
                waypoint_arm_angle = angle_created_by(
                    start=(self.a.spacial_info.x, self.a.spacial_info.y),
                    midpoint=(next_waypoint.x, next_waypoint.y),
                    end=(next_next_waypoint.x, next_next_waypoint.y),
                )
                we_passed_the_waypoint = waypoint_arm_angle < abs(math.degrees(90))
                if we_passed_the_waypoint:
                    closest_relative_index  = 1
                    
        if closest_relative_index > 0:
            self.a.next_waypoint_index += 1
            # prevent indexing error
            self.a.next_waypoint_index = min(self.a.next_waypoint_index, len(self.a.waypoints_list)-1)
            next_waypoint = self.a.waypoints_list[self.a.next_waypoint_index]
        
        self.a.closest_distance = get_distance(
            next_waypoint.x,
            next_waypoint.y,
            self.a.spacial_info.x,
            self.a.spacial_info.y
        )
        
        output, (
            _,
            self.a.crosstrack_error,
            self.a.episode_steps,
            self.a.omega_reward,
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
            relative_action=self.a.relative_action,
            absolute_action=self.a.absolute_action,
            next_waypoint_index=self.a.next_waypoint_index,
            crosstrack_error=self.a.crosstrack_error,
            action_duration=self.a.action_duration,
            ep_poses=self.a.ep_poses,
            episode_steps=self.a.episode_steps,
            horizon=self.a.horizon,
            max_number_of_timesteps_per_episode=self.a.max_number_of_timesteps_per_episode,
            number_of_waypoints=self.a.number_of_waypoints,
            omega_reward=self.a.omega_reward,
            phi_error=self.a.phi_error,
            pose=self.a.pose,
            prev_absolute_action=self.a.prev_absolute_action,
            prev_angle=self.a.prev_angle,
            reward=self.a.reward,
            total_ep_reward=self.a.total_episode_reward,
            twist=self.a.twist,
            velocity_error=self.a.velocity_error,
            velocity_reward=self.a.velocity_reward,
            waypoints_list=self.a.waypoints_list,
        )
        self.render()
        return output
    
    def reset(self, override_next_spacial_info=None):
        self.a.is_episode_start = 1
        self.a.ep_poses = []
        self.a.total_episode_reward = 0
        
        index = config.simulator.starting_waypoint
        if config.simulator.starting_waypoint == 'random':
            assert self.a.number_of_waypoints > 21
            index = np.random.randint(self.a.number_of_waypoints-20, size=1)[0]
            
        # if position is overriden by (most likely) the real world position
        if type(override_next_spacial_info) != type(None):
            # this is when the spacial_info is coming from the real world
            self.a.spacial_info = override_next_spacial_info
            self.a.next_waypoint_index = index
            self.a.prev_next_waypoint_index = index
        # simulator position
        else:
            waypoint = self.a.waypoints_list[index]
            self.a.next_waypoint_index = index
            self.a.prev_next_waypoint_index = index
            
            self.a.spacial_info = SpacialInformation(
                x=waypoint.x + config.simulator.random_start_position_offset,
                y=waypoint.y + config.simulator.random_start_position_offset,
                angle=waypoint.angle + config.simulator.random_start_angle_offset,
                velocity=0,
                spin=0,
                timestep=0,
            )
            self.a.x_pose = [self.a.spacial_info.x] * self.b.number_of_trajectories
            self.a.y_pose = [self.a.spacial_info.y] * self.b.number_of_trajectories
            for desired_velocity, waypoint in zip(self.a.desired_velocities, self.a.waypoints_list):
                waypoint.velocity = desired_velocity
        
        self.a.next_waypoint_index = index
        self.a.prev_next_waypoint_index = index
        self.a.pose = PoseEntry(
            x=float(self.a.waypoints_list[index][0] + 0.1),
            y=float(self.a.waypoints_list[index][1] + 0.1),
            angle=float(self.a.waypoints_list[index][2] + 0.01),
        )
        self.a.x_pose = [self.a.pose[0]] * self.a.number_of_trajectories
        self.a.y_pose = [self.a.pose[1]] * self.a.number_of_trajectories
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
        obs, self.a.closest_distance, self.a.next_waypoint_index = pure_get_observation(
            next_waypoint_index=self.a.next_waypoint_index,
            horizon=self.a.horizon,
            number_of_waypoints=self.a.number_of_waypoints,
            pose=self.a.pose,
            twist=self.a.twist,
            waypoints_list=self.a.waypoints_list,
        )
        return obs

    def render(self, mode="human"):
        from matplotlib.patches import Rectangle
        
        # plot all the points in blue
        x = []
        y = []
        for each_x, each_y, *_ in self.a.waypoints_list:
            x.append(each_x)
            y.append(each_y)
        self.a.ax.plot(x, y, "+b")
        
        # plot remaining points in red
        x = []
        y = []
        for each_x, each_y, *_ in self.a.waypoints_list[self.a.next_waypoint_index:]:
            x.append(each_x)
            y.append(each_y)
        self.a.ax.plot(x, y, "+r")
            
        spacial_info_x = self.a.pose[0]
        spacial_info_y = self.a.pose[1]
        spacial_info_angle = self.a.pose[2]
        
        self.a.ax.set_xlim([spacial_info_x - self.a.render_axis_size / 2.0, spacial_info_x + self.a.render_axis_size / 2.0])
        self.a.ax.set_ylim([spacial_info_y - self.a.render_axis_size / 2.0, spacial_info_y + self.a.render_axis_size / 2.0])
        total_diag_ang = self.a.diagonal_angle + spacial_info_angle
        xl = spacial_info_x - self.a.warthog_diag * math.cos(total_diag_ang)
        yl = spacial_info_y - self.a.warthog_diag * math.sin(total_diag_ang)
        self.a.rect.remove()
        self.a.rect = Rectangle(
            xy=(xl, yl), 
            width=config.vehicle.render_width * 2, 
            height=config.vehicle.render_length * 2, 
            angle=180.0 * spacial_info_angle / math.pi,
            facecolor="blue",
        )
        self.a.text.remove()
        omega_reward = -2 * math.fabs(self.b.original_relative_spin)
        self.a.text = self.a.ax.text(
            spacial_info_x + 1,
            spacial_info_y + 2,
            f"remaining_waypoints={len(self.a.waypoints_list[self.a.next_waypoint_index:])},\nvelocity_error={self.a.velocity_error:.3f}\nnext_waypoint_index={self.a.next_waypoint_index}\ncrosstrack_error={self.a.crosstrack_error:.3f}\nReward={self.a.reward:.4f}\nphi_error={self.a.phi_error*180/math.pi:.4f}\nsim step={time.time() - self.a.prev_render_timestamp:.4f}\nep_reward={self.a.total_episode_reward:.4f}\n\nomega_reward={omega_reward:.4f}\nvelocity_reward={self.a.velocity_error:.4f}",
            style="italic",
            bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
            fontsize=10,
        )
        self.a.prev_render_timestamp = time.time()
        self.a.ax.add_artist(self.a.rect)
        self.a.x_pose.append(spacial_info_x)
        self.a.y_pose.append(spacial_info_y)
        del self.a.x_pose[0]
        del self.a.y_pose[0]
        self.a.cur_pos.set_xdata(self.a.x_pose)
        self.a.cur_pos.set_ydata(self.a.y_pose)
        self.a.fig.canvas.draw()
        self.a.fig.canvas.flush_events()
        self.a.fig.savefig(f'{self.a.render_path}/{self.a.global_timestep}.png')

if not grug_test.fully_disable and (grug_test.replay_inputs or grug_test.record_io):
    @grug_test(skip=False)
    def smoke_test_warthog(out_trajectory_file):
        actual_starting_setting = config.simulator.starting_waypoint
        config.simulator.starting_waypoint = 0 # force override it for test
        try:
            env = WarthogEnv(path_to.waypoints_folder+f"/{out_trajectory_file}")
            env.should_render = False
            outputs = []
            def env_snapshot(env):
                return deepcopy(dict(
                        waypoints_list=                                getattr(env._, "waypoints_list"                                , None),
                        spacial_info=                                  getattr(env._, "spacial_info"                                  , None),
                        pose=                                          getattr(env._, "pose"                                          , None),
                        twist=                                         getattr(env._, "twist"                                         , None),
                        next_waypoint_index=                           getattr(env._, "next_waypoint_index"                           , None),
                        prev_next_waypoint_index=                      getattr(env._, "prev_next_waypoint_index"                            , None),
                        closest_distance=                              getattr(env._, "closest_distance"                              , None),
                        number_of_waypoints=                           getattr(env._, "number_of_waypoints"                           , None),
                        horizon=                                       getattr(env._, "horizon"                                       , None),
                        action_duration=                               getattr(env._, "action_duration"                               , None),
                        desired_velocities=                            getattr(env._, "desired_velocities"                            , None),
                        max_vel=                                       getattr(env._, "max_vel"                                       , None),
                        waypoints_dist=                                getattr(env._, "waypoints_dist"                                , None),
                        warthog_length=                                getattr(env._, "warthog_length"                                , None),
                        warthog_width=                                 getattr(env._, "warthog_width"                                 , None),
                        warthog_diag=                                  getattr(env._, "warthog_diag"                                  , None),
                        diag_angle=                                    getattr(env._, "diag_angle"                                    , None),
                        prev_angle=                                    getattr(env._, "prev_angle"                                    , None),
                        number_of_trajectories=                        getattr(env._, "number_of_trajectories"                                        , None),
                        x_pose=                                        getattr(env._, "x_pose"                                        , None),
                        y_pose=                                        getattr(env._, "y_pose"                                        , None),
                        crosstrack_error=                              getattr(env._, "crosstrack_error"                              , None),
                        velocity_error=                                getattr(env._, "velocity_error"                                     , None),
                        phi_error=                                     getattr(env._, "phi_error"                                     , None),
                        start_step_for_sup_data=                       getattr(env._, "start_step_for_sup_data"                       , None),
                        episode_steps=                                 getattr(env  , "episode_steps"                                 , None),
                        max_number_of_timesteps_per_episode=           getattr(env._, "max_number_of_timesteps_per_episode"           , None),
                        total_ep_reward=                               getattr(env._, "total_ep_reward"                               , None),
                        reward=                                        getattr(env._, "reward"                                        , None),
                        original_relative_velocity=                    getattr(env._, "original_relative_velocity"                    , None),
                        original_relative_spin=                        getattr(env._, "original_relative_spin"                        , None),
                        action=                                        getattr(env._, "action"                                        , None),
                        absolute_action=                               getattr(env._, "absolute_action"                               , None),
                        reaction=                                      getattr(env._, "reaction"                                      , None),
                        prev_absolute_action=                          getattr(env._, "prev_absolute_action"                          , None),
                        action_buffer=                                 getattr(env._, "action_buffer"                                 , None),
                        prev_reaction=                                 getattr(env._, "prev_reaction"                                 , None),
                        omega_reward=                                  getattr(env._, "omega_reward"                                  , None),
                        velocity_reward=                               getattr(env._, "velocity_reward"                                    , None),
                        is_delayed_dynamics=                           getattr(env._, "is_delayed_dynamics"                           , None),
                        delay_steps=                                   getattr(env._, "delay_steps"                                   , None),
                        v_delay_data=                                  getattr(env._, "v_delay_data"                                  , None),
                        w_delay_data=                                  getattr(env._, "w_delay_data"                                  , None),
                        save_data=                                     getattr(env._, "save_data"                                     , None),
                        is_episode_start=                              getattr(env._, "is_episode_start"                              , None),
                        ep_dist=                                       getattr(env._, "ep_dist"                                       , None),
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
