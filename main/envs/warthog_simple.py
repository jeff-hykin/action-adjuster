import math
import time
import csv
from copy import deepcopy
from collections import namedtuple

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
    next_waypoint_index_,
    horizon,
    number_of_waypoints,
    pose,
    twist,
    waypoints_list,
    **kwargs,
):
    obs   = [0] * (horizon * 4 + 2)
    index   = next_waypoint_index_
    
    closest_distance = math.inf
    for i in range(next_waypoint_index_, number_of_waypoints):
        dist = get_distance(x1=waypoints_list[i][0], y1=waypoints_list[i][1], x2=pose[0], y2=pose[1])
        if dist <= closest_distance:
            closest_distance = dist
            index = i
        else:
            break
    next_waypoint_index_ = index
    
    j = 0
    for i in range(0, horizon):
        k = i + next_waypoint_index_
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
    vel_error = waypoint_velocity - twist[0]
    crosstrack_error = closest_distance * math.sin(yaw_error)
    reward = (
        (2.0 - math.fabs(crosstrack_error))
        * (4.5 - math.fabs(vel_error))
        * (math.pi / 3.0 - math.fabs(phi_error))
        - math.fabs(action[0] - prev_absolute_action[0])
        - 2 * math.fabs(action[1])
    )
    if waypoint_velocity >= 2.5 and math.fabs(vel_error) > 1.5:
        reward = 0
    elif waypoint_velocity < 2.5 and math.fabs(vel_error) > 0.5:
        reward = 0
    
    return RewardOutput(reward, vel_error, crosstrack_error, phi_error)

@grug_test(max_io=5, skip=False)
def pure_reward_wrapper(
    total_ep_reward,
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
    vel_reward    = -math.fabs(action[0] - prev_absolute_action[0])
    
    reward, vel_error, crosstrack_error, phi_error = pure_reward(
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
    
    total_ep_reward = total_ep_reward + reward
    prev_absolute_action = action
    
    return reward, crosstrack_error, xdiff, ydiff, yaw_error, phi_error, vel_error, done, episode_steps, omega_reward, vel_reward, prev_absolute_action, total_ep_reward


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
    total_ep_reward,
    twist,
    vel_error,
    vel_reward,
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
    
    reward, crosstrack_error, xdiff, ydiff, yaw_error, phi_error, vel_error, done, episode_steps, omega_reward, vel_reward, prev_absolute_action, total_ep_reward = pure_reward_wrapper(
        total_ep_reward=total_ep_reward,
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
            total_ep_reward,
            vel_error,
            vel_reward,
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
    
    render_axis_size = 20
    
    def __init__(self, waypoint_file_path, trajectory_output_path=None, recorder=None, *args, **kwargs):
        super(WarthogEnv, self).__init__()
        self._ = LazyDict()
        self.log = LazyDict()
        if True:
            self.waypoint_file_path = waypoint_file_path
            self.out_trajectory_file = trajectory_output_path
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
            
            self._.max_number_of_timesteps_per_episode      = config.simulator.max_number_of_timesteps_per_episode
            self._.save_data              = config.simulator.save_data
            self._.action_duration        = config.simulator.action_duration  
            self.number_of_trajectories = config.simulator.number_of_trajectories
            self.render_axis_size       = 20
            self.next_waypoint_index    = 0
            self._.prev_next_waypoint_index_     = 0
            self._.closest_distance       = math.inf
            self._.desired_velocities     = []
            self.episode_steps          = 0
            self._.total_ep_reward        = 0
            self._.reward                 = 0
            self._.original_relative_spin           = 0 
            self.original_relative_velocity       = 0 
            self.prev_original_relative_spin      = 0 
            self.prev_original_relative_velocity  = 0 # "original" is what the actor said to do
            self.mutated_relative_spin            = 0 # "mutated" is after adversity+noise was added
            self.mutated_relative_velocity        = 0
            self.prev_mutated_relative_spin       = 0
            self.prev_mutated_relative_velocity   = 0
            self.prev_observation        = None
            self.is_episode_start        = 1
            self.trajectory_file         = None
            self._.global_timestep         = 0
            self.action_buffer           = [ (0,0) ] * config.simulator.action_delay # seed the buffer with delays
            self.simulated_battery_level = 1.0 # proportion 
            
            if self.waypoint_file_path is not None:
                self._.desired_velocities, self.waypoints_list = read_waypoint_file(self.waypoint_file_path)
            
            self._.x_pose = [0.0] * self.number_of_trajectories
            self._.y_pose = [0.0] * self.number_of_trajectories
            self._.crosstrack_error = 0
            self.velocity_error   = 0
            self._.phi_error        = 0
            self._.prev_timestamp   = time.time()
            self._.should_render    = config.simulator.should_render and countdown(config.simulator.render_rate)
        
        self.waypoints_list = []
        self._.pose = PoseEntry(
            x=0,
            y=0,
            angle=0,
        )
        self._.twist = TwistEntry(
            velocity=0,
            spin=0,
            unknown=0,
        )
        self._.next_waypoint_index_ = 0
        self._.prev_next_waypoint_index_ = 0
        self._.closest_distance = math.inf
        self._.horizon = 10
        self._.action_duration = 0.06
        self._.desired_velocities, self.waypoints_list = read_waypoint_file(waypoint_file_path)
        self._.number_of_waypoints = len(self.waypoints_list)
        self._.max_vel = 1
        self._.waypoints_dist = 0.5
        self._.warthog_length = 0.5 / 2.0
        self._.warthog_width = 1.0 / 2.0
        self._.warthog_diag = math.sqrt(
            self._.warthog_width**2 + self._.warthog_length**2
        )
        self._.diag_angle              = math.atan2(self._.warthog_length, self._.warthog_width)
        self._.prev_angle              = 0
        self._.n_traj                  = 100
        self._.x_pose                   = [0.0] * self._.n_traj
        self._.y_pose                   = [0.0] * self._.n_traj
        self._.crosstrack_error        = 0
        self._.vel_error               = 0
        self._.phi_error               = 0
        self._.start_step_for_sup_data = 500000
        self._.max_number_of_timesteps_per_episode            = 700
        self._.tprev                   = time.time()
        self._.total_ep_reward         = 0
        self._.reward                  = 0
        self._.action                  = [0.0, 0.0]
        self._.absolute_action         = [0.0, 0.0]
        self._.prev_absolute_action             = [0.0, 0.0]
        self._.omega_reward            = 0
        self._.vel_reward              = 0
        self._.delay_steps             = 5
        self._.save_data               = False
        self._.ep_dist                 = 0
        self._.ep_poses                = []
        
        self._.should_render = True
        self._.global_timestep = 0
        self._.original_relative_spin = 0 
        
        if self._.should_render:
            from matplotlib.patches import Rectangle
            self._.warthog_diag   = math.sqrt(config.vehicle.render_width**2 + config.vehicle.render_length**2)
            self._.diagonal_angle = math.atan2(config.vehicle.render_length, config.vehicle.render_width)
            self._.prev_timestamp = time.time()
            
            self._.render_path = f"{config.output_folder}/render/"
            print(f'''rendering to: {self._.render_path}''')
            FS.remove(self._.render_path)
            FS.ensure_is_folder(self._.render_path)
            plt.ion
            self._.fig = plt.figure(dpi=100, figsize=(10, 10))
            self._.ax  = self._.fig.add_subplot(111)
            self._.ax.set_xlim([-4, 4])
            self._.ax.set_ylim([-4, 4])
            self._.rect = Rectangle((0.0, 0.0), config.vehicle.render_width * 2, config.vehicle.render_length * 2, fill=False)
            self._.ax.add_artist(self._.rect)
            (self._.cur_pos,) = self._.ax.plot(self._.x_pose, self._.y_pose, "+g")
            self._.text = self._.ax.text(1, 2, f"vel_error={self._.vel_error}", style="italic", bbox={"facecolor": "red", "alpha": 0.5, "pad": 10}, fontsize=12)
            x = []
            y = []
            for each_x, each_y, *_ in self.waypoints_list:
                x.append(each_x)
                y.append(each_y)
            self._.ax.plot(x, y, "+b")
        
        self.reset()
    
    def __del__(self):
        if self.trajectory_file:
            self.trajectory_file.close()
            
    # just a wrapper around the pure_step
    def step(self, action, override_next_spacial_info=None):
        #  
        # push new action
        # 
        self.prev_original_relative_velocity = self.original_relative_velocity
        self.prev_original_relative_spin     = self._.original_relative_spin
        self.prev_mutated_relative_velocity  = self.mutated_relative_velocity
        self.prev_mutated_relative_spin      = self.mutated_relative_spin
        self.original_relative_velocity, self._.original_relative_spin = action
        self._.absolute_action = Action(
            velocity=clip(self.original_relative_velocity, min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity) * config.vehicle.controller_max_velocity,
            spin=clip(self._.original_relative_spin    , min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    ) * config.vehicle.controller_max_spin,
        )
        
        # 
        # logging and counter-increments
        # 
        if self._.save_data and self.trajectory_file is not None:
            self.trajectory_file.writelines(f"{self.spacial_info.x}, {self.spacial_info.y}, {self.spacial_info.angle}, {self.spacial_info.velocity}, {self.spacial_info.spin}, {self.original_relative_velocity}, {self._.original_relative_spin}, {self.is_episode_start}\n")
        self._.global_timestep += 1
        self.episode_steps = self.episode_steps + 1
        self.is_episode_start = 0
        
        # 
        # modify action
        # 
        if True:
            # first force them to be within normal ranges
            mutated_relative_velocity_action = clip(self.original_relative_velocity,  min=WarthogEnv.min_relative_velocity, max=WarthogEnv.max_relative_velocity)
            mutated_relative_spin_action     = clip(self._.original_relative_spin    ,  min=WarthogEnv.min_relative_spin    , max=WarthogEnv.max_relative_spin    )
            
            # 
            # ADVERSITY
            # 
            if True:
                # battery adversity
                if config.simulator.battery_adversity_enabled:
                    self.simulated_battery_level *= 1-config.simulator.battery_decay_rate
                    self.recorder.add(timestep=self._.global_timestep, simulated_battery_level=self.simulated_battery_level)
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
            # FIXME:
            # self.spacial_info = generate_next_spacial_info(
            #     old_spacial_info=SpacialInformation(*self.spacial_info),
            #     relative_velocity=self.mutated_relative_velocity,
            #     relative_spin=self.mutated_relative_spin,
            #     action_duration=self._.action_duration,
            # )
            pass
            
        self._.global_timestep += 1
        self._.action = action
        self._.relative_action = Action(velocity=self.original_relative_velocity, spin=self._.original_relative_spin) 
        
        self.spacial_info = SpacialInformation(
            x=self._.pose.x,
            y=self._.pose.y,
            angle=self._.pose.angle,
            velocity=self._.twist.velocity,
            spin=self._.twist.spin,
            timestep=self.episode_steps,
        )
        
        self._.twist, self._.prev_angle, self._.pose, ep_poses, _ = generate_next_spacial_info(
            old_spacial_info=self.spacial_info,
            relative_velocity=self._.relative_action.velocity,
            relative_spin=self._.relative_action.spin,
            action_duration=self._.action_duration,
            ep_poses=self._.ep_poses,
        )
        
        self.spacial_info = SpacialInformation(
            x=self._.pose.x,
            y=self._.pose.y,
            angle=self._.pose.angle,
            velocity=self._.twist.velocity,
            spin=self._.twist.spin,
            timestep=self.episode_steps,
        )
        
        closest_relative_index = 0
        last_waypoint_index = len(self.waypoints_list)-1
        next_waypoint          = self.waypoints_list[self.next_waypoint_index]
        if len(self.waypoints_list) > 1:
            distance_to_waypoint       = get_distance(next_waypoint.x, next_waypoint.y, self.spacial_info.x, self.spacial_info.y)
            got_further_away           = self._.closest_distance < distance_to_waypoint
            was_within_waypoint_radius = min(distance_to_waypoint, self._.closest_distance) < config.simulator.waypoint_radius
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
        
        self._.closest_distance = get_distance(
            next_waypoint.x,
            next_waypoint.y,
            self.spacial_info.x,
            self.spacial_info.y
        )
        
        output, (
            _,
            self._.crosstrack_error,
            self.episode_steps,
            self._.omega_reward,
            self._.phi_error,
            self._.prev_absolute_action,
            self._.prev_next_waypoint_index_,
            self._.reward,
            self._.total_ep_reward,
            self._.vel_error,
            self._.vel_reward,
            _,
            _,
            _,
            _,
            self._.next_waypoint_index_,
            self._.ep_poses,
        ), other = pure_step(
            relative_action=self._.relative_action,
            absolute_action=self._.absolute_action,
            next_waypoint_index_=self._.next_waypoint_index_,
            crosstrack_error=self._.crosstrack_error,
            action_duration=self._.action_duration,
            ep_poses=self._.ep_poses,
            episode_steps=self.episode_steps,
            horizon=self._.horizon,
            max_number_of_timesteps_per_episode=self._.max_number_of_timesteps_per_episode,
            number_of_waypoints=self._.number_of_waypoints,
            omega_reward=self._.omega_reward,
            phi_error=self._.phi_error,
            pose=self._.pose,
            prev_absolute_action=self._.prev_absolute_action,
            prev_angle=self._.prev_angle,
            reward=self._.reward,
            total_ep_reward=self._.total_ep_reward,
            twist=self._.twist,
            vel_error=self._.vel_error,
            vel_reward=self._.vel_reward,
            waypoints_list=self.waypoints_list,
        )
        self.render()
        return output
    
    def reset(self, override_next_spacial_info=None):
        self.is_episode_start = 1
        self._.ep_poses = []
        self._.total_ep_reward = 0
        
        index = config.simulator.starting_waypoint
        if config.simulator.starting_waypoint == 'random':
            assert self._.number_of_waypoints > 21
            index = np.random.randint(self._.number_of_waypoints-20, size=1)[0]
            
        # if position is overriden by (most likely) the real world position
        if type(override_next_spacial_info) != type(None):
            # this is when the spacial_info is coming from the real world
            self.spacial_info = override_next_spacial_info
            self.next_waypoint_index = index
            self._.prev_next_waypoint_index_ = index
        # simulator position
        else:
            waypoint = self.waypoints_list[index]
            self.next_waypoint_index = index
            self._.prev_next_waypoint_index_ = index
            
            self.spacial_info = SpacialInformation(
                x=waypoint.x + self.random_start_position_offset,
                y=waypoint.y + self.random_start_position_offset,
                angle=waypoint.angle + self.random_start_angle_offset,
                velocity=0,
                spin=0,
                timestep=0,
            )
            self._.x_pose = [self.spacial_info.x] * self.number_of_trajectories
            self._.y_pose = [self.spacial_info.y] * self.number_of_trajectories
            for desired_velocity, waypoint in zip(self._.desired_velocities, self.waypoints_list):
                waypoint.velocity = desired_velocity
        
        self._.next_waypoint_index_ = index
        self._.prev_next_waypoint_index_ = index
        self._.pose = PoseEntry(
            x=float(self.waypoints_list[index][0] + 0.1),
            y=float(self.waypoints_list[index][1] + 0.1),
            angle=float(self.waypoints_list[index][2] + 0.01),
        )
        self._.x_pose = [self._.pose[0]] * self._.n_traj
        self._.y_pose = [self._.pose[1]] * self._.n_traj
        self._.twist = TwistEntry(
            velocity=0,
            spin=0,
            unknown=0,
        )
        # for i in range(0, self._.number_of_waypoints):
        #     if self._.desired_velocities[i] > self._.max_vel:
        #         self.waypoints_list[i][3] = self._.max_vel
        #     else:
        #         self.waypoints_list[i][3] = self._.desired_velocities[i]
        # self._.max_vel = 2
        # self._.max_vel = self._.max_vel + 1
        obs, self._.closest_distance, self._.next_waypoint_index_ = pure_get_observation(
            next_waypoint_index_=self._.next_waypoint_index_,
            horizon=self._.horizon,
            number_of_waypoints=self._.number_of_waypoints,
            pose=self._.pose,
            twist=self._.twist,
            waypoints_list=self.waypoints_list,
        )
        return obs

    def render(self, mode="human"):
        from matplotlib.patches import Rectangle
        
        # plot all the points in blue
        x = []
        y = []
        for each_x, each_y, *_ in self.waypoints_list:
            x.append(each_x)
            y.append(each_y)
        self._.ax.plot(x, y, "+b")
        
        # plot remaining points in red
        x = []
        y = []
        for each_x, each_y, *_ in self.waypoints_list[self._.next_waypoint_index_:]:
            x.append(each_x)
            y.append(each_y)
        self._.ax.plot(x, y, "+r")
            
        spacial_info_x = self._.pose[0]
        spacial_info_y = self._.pose[1]
        spacial_info_angle = self._.pose[2]
        
        self._.ax.set_xlim([spacial_info_x - self.render_axis_size / 2.0, spacial_info_x + self.render_axis_size / 2.0])
        self._.ax.set_ylim([spacial_info_y - self.render_axis_size / 2.0, spacial_info_y + self.render_axis_size / 2.0])
        total_diag_ang = self._.diagonal_angle + spacial_info_angle
        xl = spacial_info_x - self._.warthog_diag * math.cos(total_diag_ang)
        yl = spacial_info_y - self._.warthog_diag * math.sin(total_diag_ang)
        self._.rect.remove()
        self._.rect = Rectangle(
            xy=(xl, yl), 
            width=config.vehicle.render_width * 2, 
            height=config.vehicle.render_length * 2, 
            angle=180.0 * spacial_info_angle / math.pi,
            facecolor="blue",
        )
        self._.text.remove()
        omega_reward = -2 * math.fabs(self._.original_relative_spin)
        self._.text = self._.ax.text(
            spacial_info_x + 1,
            spacial_info_y + 2,
            f"remaining_waypoints={len(self.waypoints_list[self._.next_waypoint_index_:])},\nvel_error={self._.vel_error:.3f}\nnext_waypoint_index_={self._.next_waypoint_index_}\ncrosstrack_error={self._.crosstrack_error:.3f}\nReward={self._.reward:.4f}\nphi_error={self._.phi_error*180/math.pi:.4f}\nsim step={time.time() - self._.prev_timestamp:.4f}\nep_reward={self._.total_ep_reward:.4f}\n\nomega_reward={omega_reward:.4f}\nvel_reward={self._.vel_error:.4f}",
            style="italic",
            bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
            fontsize=10,
        )
        self._.prev_timestamp = time.time()
        self._.ax.add_artist(self._.rect)
        self._.x_pose.append(spacial_info_x)
        self._.y_pose.append(spacial_info_y)
        del self._.x_pose[0]
        del self._.y_pose[0]
        self._.cur_pos.set_xdata(self._.x_pose)
        self._.cur_pos.set_ydata(self._.y_pose)
        self._.fig.canvas.draw()
        self._.fig.canvas.flush_events()
        self._.fig.savefig(f'{self._.render_path}/{self._.global_timestep}.png')

if not grug_test.fully_disable and (grug_test.replay_inputs or grug_test.record_io):
    @grug_test(skip=False)
    def smoke_test_warthog(trajectory_file):
        actual_starting_setting = config.simulator.starting_waypoint
        config.simulator.starting_waypoint = 0 # force override it for test
        try:
            env = WarthogEnv(path_to.waypoints_folder+f"/{trajectory_file}")
            env.should_render = False
            outputs = []
            def env_snapshot(env):
                return deepcopy(dict(
                        waypoints_list=                                getattr(env._, "waypoints_list"                                , None),
                        spacial_info=                                  getattr(env  , "spacial_info"                                  , None),
                        pose=                                          getattr(env._, "pose"                                          , None),
                        twist=                                         getattr(env._, "twist"                                         , None),
                        next_waypoint_index=                           getattr(env  , "next_waypoint_index"                           , None),
                        next_waypoint_index_=                                 getattr(env._, "next_waypoint_index_"                                 , None),
                        prev_next_waypoint_index_=                            getattr(env._, "prev_next_waypoint_index_"                            , None),
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
                        n_traj=                                        getattr(env._, "n_traj"                                        , None),
                        x_pose=                                        getattr(env._, "x_pose"                                        , None),
                        y_pose=                                        getattr(env._, "y_pose"                                        , None),
                        crosstrack_error=                              getattr(env._, "crosstrack_error"                              , None),
                        vel_error=                                     getattr(env._, "vel_error"                                     , None),
                        phi_error=                                     getattr(env._, "phi_error"                                     , None),
                        start_step_for_sup_data=                       getattr(env._, "start_step_for_sup_data"                       , None),
                        episode_steps=                                 getattr(env  , "episode_steps"                                 , None),
                        max_number_of_timesteps_per_episode=           getattr(env._, "max_number_of_timesteps_per_episode"           , None),
                        total_ep_reward=                               getattr(env._, "total_ep_reward"                               , None),
                        reward=                                        getattr(env._, "reward"                                        , None),
                        original_relative_velocity=                    getattr(env  , "original_relative_velocity"                    , None),
                        original_relative_spin=                        getattr(env._, "original_relative_spin"                        , None),
                        action=                                        getattr(env._, "action"                                        , None),
                        absolute_action=                               getattr(env._, "absolute_action"                               , None),
                        reaction=                                      getattr(env  , "reaction"                                      , None),
                        prev_absolute_action=                          getattr(env._, "prev_absolute_action"                          , None),
                        action_buffer=                                 getattr(env  , "action_buffer"                                 , None),
                        prev_reaction=                                 getattr(env  , "prev_reaction"                                 , None),
                        omega_reward=                                  getattr(env._, "omega_reward"                                  , None),
                        vel_reward=                                    getattr(env._, "vel_reward"                                    , None),
                        is_delayed_dynamics=                           getattr(env  , "is_delayed_dynamics"                           , None),
                        delay_steps=                                   getattr(env._, "delay_steps"                                   , None),
                        v_delay_data=                                  getattr(env  , "v_delay_data"                                  , None),
                        w_delay_data=                                  getattr(env  , "w_delay_data"                                  , None),
                        save_data=                                     getattr(env._, "save_data"                                     , None),
                        is_episode_start=                              getattr(env  , "is_episode_start"                              , None),
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
