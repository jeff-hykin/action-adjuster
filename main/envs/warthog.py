from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import gym
import numpy
import numpy as np
import math
from gym import spaces
import csv
import time
from config import config
from blissful_basics import Csv, create_named_list_class

# Questions:
    # what is: phi (currently hardcoded to 0)

magic_number_4         = 4.0
magic_number_2_point_5 = 2.5
magic_number_2_point_0 = 2.0
magic_number_1_point_5 = 1.5
magic_number_0_point_5 = 0.5


class WarthogEnv(gym.Env):
    warthog_length               = config.warthog.length # 0.5 / 2.0 # TODO: length in meters? why the division?
    warthog_width                = config.warthog.width  # 1.0 / 2.0  # TODO: length in meters?
    random_start_position_offset = config.simulator.random_start_position_offset
    random_start_angle_offset    = config.simulator.random_start_angle_offset
    
    action_space = spaces.Box(
        low=np.array(config.simulator.action_space.low),
        high=np.array(config.simulator.action_space.high),
        shape=np.array(config.simulator.action_space.low).shape,
    )
    observation_space = spaces.Box(
        low=config.simulator.observation_space.low,
        high=config.simulator.observation_space.high,
        shape=config.simulator.observation_space.shape,
        dtype=float,
    )
    
    SpacialInformation = create_named_list_class([ "x", "y", "angle", "velocity", "spin", ])
    
    def __init__(self, waypoint_file_path, trajectory_output_path, render=False):
        super(WarthogEnv, self).__init__()
        self.waypoint_file_path = waypoint_file_path
        self.out_trajectory_file = trajectory_output_path
        
        self.waypoints_list   = []
        self.spacial_info = WarthogEnv.SpacialInformation([])
        self.spacial_info.x                = 0
        self.spacial_info.y                = 0
        self.spacial_info.angle            = 0
        self.spacial_info.velocity         = 0
        self.spacial_info.spin             = 0
        
        self.max_episode_steps      = config.simulator.max_episode_steps
        self.save_data              = config.simulator.save_data
        self.dt                     = config.simulator.dt  # TODO: why is dt 0.06?
        self.use_delayed_dynamics   = config.simulator.use_delayed_dynamics
        self.delay_steps            = config.simulator.delay_steps
        self.horizon                = config.simulator.horizon # number of waypoints in the observation
        self.number_of_trajectories = config.simulator.number_of_trajectories
        self.render_axis_size       = 20
        self.closest_index          = 0
        self.prev_closest_index     = 0
        self.closest_distance       = math.inf
        self.desired_velocities     = []
        self.max_velocity           = 1  # TODO: currently increases over time, not sure if thats intended
        self.episode_steps          = 0
        self.total_episode_reward   = 0
        self.reward                 = 0
        self.action_spin            = 0
        self.action_velocity        = 0
        self.prev_action_spin       = 0
        self.prev_action_velocity   = 0
        self.omega_reward           = 0
        self.velocity_reward        = 0
        self.velocity_delay_data    = [0.0] * self.delay_steps
        self.spin_delay_data        = [0.0] * self.delay_steps
        self.is_episode_start       = 1
        self.trajectory_file        = None
        
        if self.waypoint_file_path is not None:
            self._read_waypoint_file_path(self.waypoint_file_path)
        
        self.warthog_diag   = math.sqrt(self.warthog_width**2 + self.warthog_length**2)
        self.diagonal_angle = math.atan2(self.warthog_length, self.warthog_width)
        self.prev_angle = 0
        self.x_pose = [0.0] * self.number_of_trajectories
        self.y_pose = [0.0] * self.number_of_trajectories
        self.crosstrack_error = 0
        self.velocity_error   = 0
        self.phi_error        = 0
        self.prev_timestamp   = time.time()
        
        if render:
            plt.ion
            self.fig = plt.figure(dpi=100, figsize=(10, 10))
            self.ax  = self.fig.add_subplot(111)
            self.ax.set_xlim([-4, 4])
            self.ax.set_ylim([-4, 4])
            self.rect = Rectangle((0.0, 0.0), self.warthog_width * 2, self.warthog_length * 2, fill=False)
            self.ax.add_artist(self.rect)
            (self.cur_pos,) = self.ax.plot(self.x_pose, self.y_pose, "+g")
            self.text = self.ax.text(1, 2, f"vel_error={self.velocity_error}", style="italic", bbox={"facecolor": "red", "alpha": 0.5, "pad": 10}, fontsize=12)
            if self.waypoint_file_path is not None:
                self.plot_waypoints()
        
        # 
        # trajectory_file
        # 
        if self.out_trajectory_file is not None:
            self.trajectory_file = open(trajectory_output_path, "w")
    
    @property
    def number_of_waypoints(self):
        return len(self.waypoints_list)
    
    def __del__(self):
        if self.trajectory_file:
            self.trajectory_file.close()

    def plot_waypoints(self):
        x = []
        y = []
        for each_waypoint in self.waypoints_list:
            x.append(each_waypoint.x)
            y.append(each_waypoint.y)
        self.ax.plot(x, y, "+r")

    def sim_warthog(self, old_spatial_info, velocity_action, spin_action):
        velocity_action = np.clip(velocity_action,  0, 1) * magic_number_4, # TODO: is it supposed to be clipped to self.max_velocity?
        spin_action     = np.clip(spin_action,     -1, 1) * magic_number_2_point_5,
        
        old_velocity = old_spatial_info.velocity
        old_spin     = old_spatial_info.spin
        old_x        = old_spatial_info.x
        old_y        = old_spatial_info.y
        old_angle    = old_spatial_info.angle
        
        new_spacial_info = WarthogEnv.SpacialInformation(old_spatial_info)
        
        new_spacial_info.velocity = velocity
        new_spacial_info.spin     = spin
        
        if self.use_delayed_dynamics:
            old_velocity = self.velocity_delay_data[0]
            old_spin     = self.spin_delay_data[0]
            del self.velocity_delay_data[0]
            del self.spin_delay_data[0]
            self.velocity_delay_data.append(velocity)
            self.spin_delay_data.append(spin)
            new_spacial_info.velocity = self.velocity_delay_data[0]
            new_spacial_info.spin     = self.spin_delay_data[0] # TODO: I dont think this old version worked -> new_spacial_info.spin     = self.velocity_delay_data[1]
        
        self.prev_angle = old_angle
        new_spacial_info.x          = old_x + old_velocity * math.cos(old_angle) * self.dt
        new_spacial_info.y          = old_y + old_velocity * math.sin(old_angle) * self.dt
        new_spacial_info.angle      = zero_to_2pi(old_angle + old_spin * self.dt)
        if self.save_data and self.trajectory_file is not None:
            self.trajectory_file.writelines(f"{old_x}, {old_y}, {old_angle}, {old_velocity}, {old_spin}, {velocity}, {spin}, {self.is_episode_start}\n")
        self.is_episode_start = 0

    def update_closest_index(self, x, y):
        closest_index = self.closest_index
        self.closest_distance = math.inf
        for index in range(self.closest_index, self.number_of_waypoints):
            waypoint = self.waypoints_list[index]
            distance = get_distance(waypoint.x, waypoint.y, x, y)
            if distance <= self.closest_distance:
                self.closest_distance = distance
                closest_index = index
            else:
                break
        self.closest_index = closest_index

    def get_observation(self):
        magic_number_4 = 4 # I think this is len([x,y,spin,velocity])
        magic_number_2 = 2 # TODO: I have no idea what this 2 is for
        obs = [0] * ((self.horizon * magic_number_4) + magic_number_2)
        original_velocity = self.spacial_info.velocity
        original_spin     = self.spacial_info.spin
        self.update_closest_index(x=self.spacial_info.x, y=self.spacial_info.y)
        observation_index = 0
        for horizon_index in range(0, self.horizon):
            waypoint_index = horizon_index + self.closest_index
            if waypoint_index < self.number_of_waypoints:
                waypoint = self.waypoints_list[waypoint_index]
                x_diff = waypoint.x - self.spacial_info.x
                y_diff = waypoint.y - self.spacial_info.y
                radius = get_distance(waypoint.x, waypoint.y, self.spacial_info.x, self.spacial_info.y)
                angle_to_next_point = get_angle_from_origin(x_diff, y_diff)
                current_angle       = zero_to_2pi(self.spacial_info.angle)
                
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

    def step(self, action):
        self.episode_steps = self.episode_steps + 1
        self.action_velocity, self.action_spin = action
        
        # 
        # add noise
        # 
        velocity_noise = 0
        spin_noise     = 0
        if config.simulator.use_gaussian_action_noise:
            import random
            velocity_noise = self.action_velocity - random.normalvariate(
                mu=self.action_velocity,
                sigma=config.simulator.gaussian_action_noise.velocity_action.standard_deviation,
            )
            spin_noise = self.action_spin - random.normalvariate(
                mu=self.action_spin,
                sigma=config.simulator.gaussian_action_noise.spin_action.standard_deviation,
            )
        
        self.sim_warthog(
            old_spatial_info=WarthogEnv.SpacialInformation(self.spacial_info),
            velocity_action=self.action_velocity + velocity_noise,
            spin_action=self.action_spin + spin_noise,
        )
        self.prev_closest_index = self.closest_index
        observation = self.get_observation()
        done = False
        if self.closest_index >= self.number_of_waypoints - 1:
            done = True
        
        # 
        # Reward Calculation
        # 
        closest_waypoint = self.waypoints_list[self.closest_index]
        
        x_diff     = closest_waypoint.x - self.spacial_info.x
        y_diff     = closest_waypoint.y - self.spacial_info.y
        angle_diff = get_angle_from_origin(x_diff, y_diff)
        yaw_error  = pi_to_pi(angle_diff - self.spacial_info.angle)
        
        self.velocity_error   = closest_waypoint.velocity - self.spacial_info.velocity
        self.crosstrack_error = self.closest_distance * math.sin(yaw_error)
        self.phi_error        = pi_to_pi(closest_waypoint.angle - self.spacial_info.angle)
        if math.fabs(self.crosstrack_error) > magic_number_1_point_5 or math.fabs(self.phi_error) > 1.4:
            done = True
        if self.episode_steps == self.max_episode_steps:
            done = True
            self.episode_steps = 0
        
        magic_number_2 = 2.0
        magic_number_3 = 3.0
        magic_number_0_point_5 = 0.5
        magic_number_4_point_5 = 4.5
        
        crosstrack_reward = magic_number_2           - math.fabs(self.crosstrack_error)
        velocity_reward   = magic_number_4_point_5   - math.fabs(self.velocity_error)
        angle_reward      = math.pi / magic_number_3 - math.fabs(self.phi_error)
        
        running_reward = crosstrack_reward * velocity_reward * angle_reward
        
        # penalties
        running_reward -= math.fabs(self.action_velocity - self.prev_action_velocity)                  # velocity penalty
        running_reward -= magic_number_0_point_5 * math.fabs(self.action_spin - self.prev_action_spin) # spin penalty
        running_reward -= math.fabs(self.action_spin)                                                  # TODO: ??? penalty
        
        self.reward = running_reward

        self.omega_reward         = -2 * math.fabs(self.action_spin)
        self.velocity_reward      = -math.fabs(self.action_velocity - self.prev_action_velocity)
        self.prev_action_spin     = self.action_spin
        self.prev_action_velocity = self.action_velocity
        
        if closest_waypoint.velocity >= magic_number_2_point_5 and math.fabs(self.velocity_error) > magic_number_1_point_5:
            self.reward = 0
        elif closest_waypoint.velocity < magic_number_2_point_5 and math.fabs(self.velocity_error) > magic_number_0_point_5:
            self.reward = 0
        self.total_episode_reward = self.total_episode_reward + self.reward
        
        return observation, self.reward, done, {}

    def reset(self):
        self.is_episode_start = 1
        self.total_episode_reward = 0
        if self.max_velocity >= 5:
            self.max_velocity = 1
        
        # pick a random waypoint
        index = np.random.randint(self.number_of_waypoints, size=1)[0]
        self.closest_index      = index
        self.prev_closest_index = index
        waypoint = self.waypoints_list[index]
        
        self.spacial_info.x      = waypoint.x + self.random_start_position_offset
        self.spacial_info.y      = waypoint.y + self.random_start_position_offset
        self.spacial_info.angle  = waypoint.angle + self.random_start_angle_offset
        self.x_pose = [self.spacial_info.x] * self.number_of_trajectories
        self.y_pose = [self.spacial_info.y] * self.number_of_trajectories
        self.spacial_info.velocity = 0
        self.spacial_info.spin     = 0
        for desired_velocity, waypoint in zip(self.desired_velocities, self.waypoints_list):
            if desired_velocity > self.max_velocity:
                waypoint.velocity = self.max_velocity
            else:
                waypoint.velocity = desired_velocity
        
        self.max_velocity = self.max_velocity + 1 # TODO: check that this is right
        return self.get_observation()

    def render(self, mode="human"):
        self.ax.set_xlim([self.spacial_info.x - self.render_axis_size / 2.0, self.spacial_info.x + self.render_axis_size / 2.0])
        self.ax.set_ylim([self.spacial_info.y - self.render_axis_size / 2.0, self.spacial_info.y + self.render_axis_size / 2.0])
        total_diag_ang = self.diagonal_angle + self.spacial_info.angle
        xl = self.spacial_info.x - self.warthog_diag * math.cos(total_diag_ang)
        yl = self.spacial_info.y - self.warthog_diag * math.sin(total_diag_ang)
        self.rect.remove()
        self.rect = Rectangle((xl, yl), self.warthog_width * 2, self.warthog_length * 2, 180.0 * self.spacial_info.angle / math.pi, facecolor="blue")
        self.text.remove()
        self.text = self.ax.text(
            self.spacial_info.x + 1,
            self.spacial_info.y + 2,
            f"vel_error={self.velocity_error:.3f}\nclosest_index={self.closest_index}\ncrosstrack_error={self.crosstrack_error:.3f}\nReward={self.reward:.4f}\nwarthog_vel={self.spacial_info.velocity:.3f}\nphi_error={self.phi_error*180/math.pi:.4f}\nsim step={time.time() - self.prev_timestamp:.4f}\nep_reward={self.total_episode_reward:.4f}\nmax_vel={self.max_velocity:.4f}\nomega_reward={self.omega_reward:.4f}\nvel_reward={self.velocity_error:.4f}",
            style="italic",
            bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
            fontsize=10,
        )
        self.prev_timestamp = time.time()
        self.ax.add_artist(self.rect)
        self.x_pose.append(self.spacial_info.x)
        self.y_pose.append(self.spacial_info.y)
        del self.x_pose[0]
        del self.y_pose[0]
        self.cur_pos.set_xdata(self.x_pose)
        self.cur_pos.set_ydata(self.y_pose)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _read_waypoint_file_path(self, filename):
        for row in Csv.read(filename, seperator=",", first_row_is_column_names=True, skip_empty_lines=True):
            self.desired_velocities.append(row.velocity)
            self.waypoints_list.append(
                Waypoint([row.x, row.y, row.angle, row.velocity])
            )
        
        index = 0
        for index in range(0, len(self.waypoints_list) - 1):
            current_waypoint = self.waypoints_list[index]
            next_waypoint    = self.waypoints_list[index+1]
            
            x_diff = next_waypoint.x - current_waypoint.x
            y_diff = next_waypoint.y - current_waypoint.y
            current_waypoint.angle = zero_to_2pi(get_angle_from_origin(x_diff, y_diff))
        
        assert self.waypoints_list[index+1].tolist() == self.waypoints_list[-1].tolist()
        final_waypoint = self.waypoints_list[index + 1]
        final_waypoint.angle = self.waypoints_list[-2].angle # fill in the blank value

def get_distance(x1, y1, x2, y2):
    x_diff = x2 - x1
    y_diff = y2 - y1
    return math.sqrt(x_diff * x_diff + y_diff * y_diff)

def get_angle_from_origin(x, y):
    theta = math.atan2(y, x)
    return zero_to_2pi(theta)

def zero_to_2pi(theta):
    if theta < 0:
        theta = 2 * math.pi + theta
    elif theta > 2 * math.pi:
        theta = theta - 2 * math.pi
    return theta

def pi_to_pi(theta):
    if theta < -math.pi:
        theta = theta + 2 * math.pi
    elif theta > math.pi:
        theta = theta - 2 * math.pi
    return theta

def Waypoint(a_list):
    waypoint_entry = WaypointEntry(len(a_list))
    for index, each in enumerate(a_list):
        waypoint_entry[index] = each
    return waypoint_entry

class WaypointEntry(numpy.ndarray):
    @property
    def x(self): return self[0]
    @x.setter
    def x(self, value): self[0] = value
    
    @property
    def y(self): return self[1]
    @y.setter
    def y(self, value): self[1] = value
    
    @property
    def angle(self): return self[2]
    @angle.setter
    def angle(self, value): self[2] = value
    
    @property
    def velocity(self): return self[3]
    @velocity.setter
    def velocity(self, value): self[3] = value
