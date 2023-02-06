from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import numpy
import numpy as np
import math
import gym
import csv
import time
from blissful_basics import Csv, create_named_list_class
import file_system_py as FS

from config import config
from tools.geometry import get_distance, get_angle_from_origin, zero_to_2pi, pi_to_pi, abs_angle_difference

max_velocity_reset_number = 5 # TODO: check this
magic_number_2_point_5 = 2.5
magic_number_2_point_0 = 2.0
magic_number_1_point_5 = 1.5
magic_number_1_point_4 = 1.4
magic_number_0_point_5 = 0.5


class WarthogEnv(gym.Env):
    random_start_position_offset = config.simulator.random_start_position_offset
    random_start_angle_offset    = config.simulator.random_start_angle_offset
    
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
    
    SpacialInformation = create_named_list_class([ "x", "y", "angle", "velocity", "spin", ])
    
    def __init__(self, waypoint_file_path, trajectory_output_path, should_render=True):
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
        self.action_duration        = config.simulator.action_duration  
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
        self.is_episode_start       = 1
        self.trajectory_file        = None
        self.global_timestep        = 0
        self.action_buffer          = [ (0,0) ] * config.simulator.action_delay # seed the buffer with delays
        
        if self.waypoint_file_path is not None:
            self._read_waypoint_file_path(self.waypoint_file_path)
        
        self.prev_angle = 0
        self.x_pose = [0.0] * self.number_of_trajectories
        self.y_pose = [0.0] * self.number_of_trajectories
        self.crosstrack_error = 0
        self.velocity_error   = 0
        self.phi_error        = 0
        self.prev_timestamp   = time.time()
        self.should_render    = should_render
        
        if self.should_render:
            self.warthog_diag   = math.sqrt(config.vehicle.render_width**2 + config.vehicle.render_length**2)
            self.diagonal_angle = math.atan2(config.vehicle.render_length, config.vehicle.render_width)
            
            self.render_path = "render.ignore/"
            FS.ensure_is_folder(self.render_path)
            plt.ion
            self.fig = plt.figure(dpi=100, figsize=(10, 10))
            self.ax  = self.fig.add_subplot(111)
            self.ax.set_xlim([-4, 4])
            self.ax.set_ylim([-4, 4])
            self.rect = Rectangle((0.0, 0.0), config.vehicle.render_width * 2, config.vehicle.render_length * 2, fill=False)
            self.ax.add_artist(self.rect)
            (self.cur_pos,) = self.ax.plot(self.x_pose, self.y_pose, "+g")
            self.text = self.ax.text(1, 2, f"vel_error={self.velocity_error}", style="italic", bbox={"facecolor": "red", "alpha": 0.5, "pad": 10}, fontsize=12)
            if self.waypoint_file_path is not None:
                self.plot_waypoints()
        
        # 
        # trajectory_file
        # 
        if self.out_trajectory_file is not None:
            FS.ensure_is_folder(FS.parent_path(trajectory_output_path))
            self.trajectory_file = open(trajectory_output_path, "w+")
            self.trajectory_file.writelines(f"x, y, angle, velocity, spin, velocity_action, spin_action, is_episode_start\n")
    
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

    @staticmethod
    def sim_warthog(old_spatial_info, velocity_action, spin_action, action_duration):
        velocity_action = np.clip(velocity_action,  0, 1) * config.vehicle.controller_max_velocity
        spin_action     = np.clip(spin_action,     -1, 1) * config.vehicle.controller_max_spin
        
        old_velocity = old_spatial_info.velocity
        old_spin     = old_spatial_info.spin
        old_x        = old_spatial_info.x
        old_y        = old_spatial_info.y
        old_angle    = old_spatial_info.angle
        
        new_spacial_info          = WarthogEnv.SpacialInformation(old_spatial_info)
        new_spacial_info.velocity = velocity_action
        new_spacial_info.spin     = spin_action
        new_spacial_info.x        = old_x + old_velocity * math.cos(old_angle) * action_duration
        new_spacial_info.y        = old_y + old_velocity * math.sin(old_angle) * action_duration
        new_spacial_info.angle    = zero_to_2pi(old_angle + old_spin           * action_duration)
        
        return new_spacial_info
    
    @staticmethod
    def get_closest_index(remaining_waypoints, x, y):
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

    @staticmethod
    def generate_observation(remaining_waypoints, horizon, current_spacial_info):
        original_velocity = current_spacial_info.velocity
        original_spin     = current_spacial_info.spin
        
        observation = []
        closest_index = WarthogEnv.get_closest_index(remaining_waypoints, current_spacial_info.x, current_spacial_info.y)
        for horizon_index in range(0, horizon):
            waypoint_index = horizon_index + closest_index
            if waypoint_index < len(remaining_waypoints):
                waypoint = remaining_waypoints[waypoint_index]
                
                x_diff = waypoint.x - current_spacial_info.x
                y_diff = waypoint.y - current_spacial_info.y
                angle_to_next_point = get_angle_from_origin(x_diff, y_diff)
                current_angle       = zero_to_2pi(current_spacial_info.angle)
                
                gap_of_distance                    = get_distance(waypoint.x, waypoint.y, current_spacial_info.x, current_spacial_info.y)
                gap_of_angle_directly_towards_next = pi_to_pi(angle_to_next_point - current_angle)
                gap_of_desired_angle_at_next       = pi_to_pi(waypoint.angle      - current_angle)
                gap_of_velocity                    = waypoint.velocity - original_velocity
                
                observation.append(gap_of_distance)
                observation.append(gap_of_angle_directly_towards_next)
                observation.append(gap_of_desired_angle_at_next)
                observation.append(gap_of_velocity)
            else:
                observation.append(0.0)
                observation.append(0.0)
                observation.append(0.0)
                observation.append(0.0)
        
        observation.append(original_velocity)
        observation.append(original_spin)
        return observation


    def step(self, action):
        self.prev_action_spin, self.prev_action_velocity = self.action_velocity, self.action_spin
        self.action_velocity, self.action_spin = action
        
        # 
        # logging and counter-increments
        # 
        if self.save_data and self.trajectory_file is not None:
            self.trajectory_file.writelines(f"{self.spacial_info.x}, {self.spacial_info.y}, {self.spacial_info.angle}, {self.spacial_info.velocity}, {self.spacial_info.spin}, {self.action_velocity}, {self.action_spin}, {self.is_episode_start}\n")
        self.global_timestep += 1
        self.episode_steps = self.episode_steps + 1
        self.is_episode_start = 0
        
        # 
        # handle action + spacial update
        # 
        if True:
            velocity_action = self.action_velocity
            spin_action     = self.action_spin
        
            # 
            # add noise
            # 
            velocity_noise = 0
            spin_noise     = 0
            if config.simulator.use_gaussian_action_noise:
                import random
                velocity_noise = self.action_velocity - random.normalvariate(mu=self.action_velocity, sigma=config.simulator.gaussian_action_noise.velocity_action.standard_deviation, )
                spin_noise     = self.action_spin     - random.normalvariate(mu=self.action_spin    , sigma=config.simulator.gaussian_action_noise.spin_action.standard_deviation    , )
            
            # 
            # action delay
            # 
            self.action_buffer.append((velocity_action, spin_action))
            velocity_action, spin_action = self.action_buffer.pop(0) # if no delay, this will pop what was just pushed
                
            # 
            # apply action
            # 
            self.spacial_info = WarthogEnv.sim_warthog(
                old_spatial_info=WarthogEnv.SpacialInformation(self.spacial_info),
                velocity_action=self.action_velocity + velocity_noise,
                spin_action=self.action_spin + spin_noise,
                action_duration=self.action_duration,
            )
        
        # 
        # observation
        # 
        if True:
            # 
            # add positional noise
            # 
            x_noise        = 0
            y_noise        = 0
            angle_noise    = 0
            spin_noise     = 0
            velocity_noise = 0
            if config.simulator.use_gaussian_spacial_noise:
                import random
                x_noise        = self.spacial_info.x        - random.normalvariate( mu=self.spacial_info.x       , sigma=config.simulator.gaussian_spacial_noise.x.standard_deviation       ,)
                y_noise        = self.spacial_info.y        - random.normalvariate( mu=self.spacial_info.y       , sigma=config.simulator.gaussian_spacial_noise.y.standard_deviation       , )
                angle_noise    = self.spacial_info.angle    - random.normalvariate( mu=self.spacial_info.angle   , sigma=config.simulator.gaussian_spacial_noise.angle.standard_deviation   , )
                spin_noise     = self.spacial_info.spin     - random.normalvariate( mu=self.spacial_info.spin    , sigma=config.simulator.gaussian_spacial_noise.spin.standard_deviation    , )
                velocity_noise = self.spacial_info.velocity - random.normalvariate( mu=self.spacial_info.velocity, sigma=config.simulator.gaussian_spacial_noise.velocity.standard_deviation, )
            
            # generate observation off potentially incorrect spacial info
            spacial_info_with_noise = WarthogEnv.SpacialInformation([
                self.spacial_info.x        + x_noise       ,
                self.spacial_info.y        + y_noise       ,
                self.spacial_info.angle    + angle_noise   ,
                self.spacial_info.velocity + velocity_noise,
                self.spacial_info.spin     + spin_noise    ,
            ])
            observation = WarthogEnv.generate_observation(
                remaining_waypoints=self.waypoints_list[self.closest_index:],
                horizon=self.horizon,
                current_spacial_info=spacial_info_with_noise,
            )
        
        
        # 
        # Reward Calculation
        # 
        if True:
            # 
            # get the true closest waypoint (e.g. perfect sensors)
            #
            self.prev_closest_index = self.closest_index 
            self.closest_index = WarthogEnv.get_closest_index(
                remaining_waypoints=self.waypoints_list[self.closest_index:],
                x=self.spacial_info.x,
                y=self.spacial_info.y,
            )
            closest_waypoint = self.waypoints_list[self.closest_index]
            
            x_diff     = closest_waypoint.x - self.spacial_info.x
            y_diff     = closest_waypoint.y - self.spacial_info.y
            angle_diff = get_angle_from_origin(x_diff, y_diff)
            yaw_error  = pi_to_pi(angle_diff - self.spacial_info.angle)
            
            self.velocity_error   = closest_waypoint.velocity - self.spacial_info.velocity
            self.crosstrack_error = self.closest_distance * math.sin(yaw_error)
            self.phi_error        = pi_to_pi(closest_waypoint.angle - self.spacial_info.angle)
            
            max_expected_crosstrack_error = config.reward_parameters.max_expected_crosstrack_error # meters
            max_expected_velocity_error   = config.reward_parameters.max_expected_velocity_error * config.vehicle.controller_max_velocity # meters per second
            max_expected_angle_error      = config.reward_parameters.max_expected_angle_error # 60° but in radians
            
            # base rewards
            crosstrack_reward = max_expected_crosstrack_error - math.fabs(self.crosstrack_error)
            velocity_reward   = max_expected_velocity_error   - math.fabs(self.velocity_error)
            angle_reward      = max_expected_angle_error      - math.fabs(self.phi_error)
            
            # combine
            running_reward = crosstrack_reward * velocity_reward * angle_reward
            
            # smoothing penalties (jerky=costly to machine and high energy/breaks consumption)
            running_reward -= config.reward_parameters.velocity_jerk_cost_coefficient * math.fabs(self.action_velocity - self.prev_action_velocity)
            running_reward -= config.reward_parameters.spin_jerk_cost_coefficient     * math.fabs(self.action_spin - self.prev_action_spin)
            # direct energy consumption
            running_reward -= config.reward_parameters.direct_velocity_cost * math.fabs(self.action_velocity)
            running_reward -= config.reward_parameters.direct_spin_cost     * math.fabs(self.action_spin)
            
            self.reward = running_reward
            
            # 
            # velocity caps
            # 
            if config.reward_parameters.velocity_caps_enabled:
                velocity_error = math.fabs(self.velocity_error)
                for min_waypoint_speed, max_error_allowed in config.reward_parameters.velocity_caps.items():
                    # convert %'s to vehicle-specific values
                    min_waypoint_speed = float(min_waypoint_speed.replace("%", ""))/100 * config.vehicle.controller_max_velocity
                    max_error_allowed  = float(max_error_allowed.replace( "%", ""))/100 * config.vehicle.controller_max_velocity
                    # if rule-is-active
                    if closest_waypoint.velocity >= min_waypoint_speed:
                        # if rule is broken, no reward
                        if velocity_error > max_error_allowed:
                            self.reward = 0
                            break
        # 
        # render
        # 
        if self.should_render:
            self.render()
        
        additional_info = dict(
            spacial_info=self.spacial_info,
            spacial_info_with_noise=spacial_info_with_noise,
            remaining_waypoints=self.waypoints_list[self.closest_index:],
            horizon=self.horizon,
            action_duration=self.action_duration,
        )
        
        # 
        # done Calculation
        #
        done = False
        if self.closest_index >= self.number_of_waypoints - 1:
            done = True 
        if self.episode_steps == self.max_episode_steps:
            done = True
            self.episode_steps = 0
        # immediate end due to too much loss
        if config.simulator.allow_cut_short_episode:
            if math.fabs(self.crosstrack_error) > magic_number_1_point_5 or math.fabs(self.phi_error) > magic_number_1_point_4:
                done = True
        
        return observation, self.reward, done, additional_info

    def reset(self):
        self.is_episode_start = 1
        self.total_episode_reward = 0
        if self.max_velocity >= max_velocity_reset_number:
            self.max_velocity = 1
        
        # pick a random waypoint
        index = np.random.randint(self.number_of_waypoints, size=1)[0]
        waypoint = self.waypoints_list[index]
        self.closest_index      = index
        self.prev_closest_index = index
        
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
        return WarthogEnv.generate_observation(
            remaining_waypoints=self.waypoints_list,
            horizon=self.horizon,
            current_spacial_info=self.spacial_info,
        )

    def render(self, mode="human"):
        self.ax.set_xlim([self.spacial_info.x - self.render_axis_size / 2.0, self.spacial_info.x + self.render_axis_size / 2.0])
        self.ax.set_ylim([self.spacial_info.y - self.render_axis_size / 2.0, self.spacial_info.y + self.render_axis_size / 2.0])
        total_diag_ang = self.diagonal_angle + self.spacial_info.angle
        xl = self.spacial_info.x - self.warthog_diag * math.cos(total_diag_ang)
        yl = self.spacial_info.y - self.warthog_diag * math.sin(total_diag_ang)
        self.rect.remove()
        self.rect = Rectangle(
            xy=(xl, yl), 
            width=config.vehicle.render_width * 2, 
            height=config.vehicle.render_length * 2, 
            angle=180.0 * self.spacial_info.angle / math.pi,
            facecolor="blue",
        )
        self.text.remove()
        omega_reward = -2 * math.fabs(self.action_spin)
        self.text = self.ax.text(
            self.spacial_info.x + 1,
            self.spacial_info.y + 2,
            f"vel_error={self.velocity_error:.3f}\nclosest_index={self.closest_index}\ncrosstrack_error={self.crosstrack_error:.3f}\nReward={self.reward:.4f}\nwarthog_vel={self.spacial_info.velocity:.3f}\nphi_error={self.phi_error*180/math.pi:.4f}\nsim step={time.time() - self.prev_timestamp:.4f}\nep_reward={self.total_episode_reward:.4f}\nmax_vel={self.max_velocity:.4f}\nomega_reward={omega_reward:.4f}\nvel_reward={self.velocity_error:.4f}",
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
        self.fig.savefig(f'{self.render_path}/{self.global_timestep}.png')


    def _read_waypoint_file_path(self, filename):
        comments, column_names, rows = Csv.read(filename, seperator=",", first_row_is_column_names=True, skip_empty_lines=True)
        for row in rows:
            self.desired_velocities.append(row.velocity)
            self.waypoints_list.append(
                Waypoint([row.x, row.y, row.angle, row.velocity])
            )
        
        index = 0
        for index in range(0, len(self.waypoints_list) - 1):
            current_waypoint = self.waypoints_list[index]
            next_waypoint    = self.waypoints_list[index+1]
            
            x_diff = next_waypoint.x - current_waypoint.x # meters
            y_diff = next_waypoint.y - current_waypoint.y # meters
            # angle radians
            current_waypoint.angle = zero_to_2pi(get_angle_from_origin(x_diff, y_diff))
        
        assert self.waypoints_list[index+1].tolist() == self.waypoints_list[-1].tolist()
        final_waypoint = self.waypoints_list[index + 1]
        final_waypoint.angle = self.waypoints_list[-2].angle # fill in the blank value


def Waypoint(a_list):
    waypoint_entry = WaypointEntry(len(a_list))
    for index, each in enumerate(a_list):
        waypoint_entry[index] = each
    return waypoint_entry

class WaypointEntry(numpy.ndarray):
    keys = [ "x", "y", "angle", "velocity" ]
    
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
