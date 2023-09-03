import math
import time
import csv

from gym import spaces
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import gym
import matplotlib as mpl
import numpy as np

class WarthogEnv(gym.Env):
    action_space = spaces.Box(
        low=np.array([0.0, -1.5]),
        high=np.array([1.0, 1.5]),
        shape=(2,)
    )
    observation_space = spaces.Box(
        low=-100,
        high=1000,
        shape=(42,),
        dtype=float,
    )
    
    def __init__(self, waypoint_file, *args, **kwargs):
        super(WarthogEnv, self).__init__()
        plt.ion
        self.waypoints_list = []
        self.pose = [0, 0, 0]
        self.twist = [0, 0]
        self.closest_index = 0
        self.prev_closest_index = 0
        self.closest_dist = math.inf
        self.number_of_waypoints = 0
        self.horizon = 10
        self.dt = 0.06
        self.ref_vel = []
        self.num_steps = 0
        self.waypoints_list, self.ref_vel, self.number_of_waypoints = read_waypoint_file(waypoint_file)
        self.max_vel = 1
        self.waypoints_dist = 0.5
        self.warthog_length = 0.5 / 2.0
        self.warthog_width = 1.0 / 2.0
        self.warthog_diag = math.sqrt(
            self.warthog_width**2 + self.warthog_length**2
        )
        self.diag_angle              = math.atan2(self.warthog_length, self.warthog_width)
        self.prev_angle              = 0
        self.n_traj                  = 100
        self.xpose                   = [0.0] * self.n_traj
        self.ypose                   = [0.0] * self.n_traj
        self.crosstrack_error        = 0
        self.vel_error               = 0
        self.phi_error               = 0
        self.start_step_for_sup_data = 500000
        self.ep_steps                = 0
        self.max_ep_steps            = 700
        self.tprev                   = time.time()
        self.total_ep_reward         = 0
        self.reward                  = 0
        self.action                  = [0.0, 0.0]
        self.prev_action             = [0.0, 0.0]
        self.omega_reward            = 0
        self.vel_reward              = 0
        self.is_delayed_dynamics     = False
        self.delay_steps             = 5
        self.v_delay_data            = [0.0] * self.delay_steps
        self.w_delay_data            = [0.0] * self.delay_steps
        self.save_data               = False
        self.ep_start                = 1
        self.ep_dist                 = 0
        self.ep_poses                = []

    def step(self, action):
        self.ep_steps = self.ep_steps + 1
        self.num_steps = self.num_steps + 1
        action[0] = np.clip(action[0], 0, 1) * 4.0
        action[1] = np.clip(action[1], -1, 1) * 2.5
        self.action = action
        self.sim_warthog(action[0], action[1])
        self.prev_closest_index = self.closest_index
        obs = self.get_observation()
        done = False
        if self.closest_index >= self.number_of_waypoints - 1:
            done = True
        
        # 
        # Calculating reward
        # 
        k = self.closest_index
        xdiff = self.waypoints_list[k][0] - self.pose[0]
        ydiff = self.waypoints_list[k][1] - self.pose[1]
        th = get_theta(xdiff, ydiff)
        yaw_error = pi_to_pi(th - self.pose[2])
        self.phi_error = pi_to_pi(
            self.waypoints_list[self.closest_index][2] - self.pose[2]
        )
        self.vel_error = self.waypoints_list[k][3] - self.twist[0]
        self.crosstrack_error = self.closest_dist * math.sin(yaw_error)
        if math.fabs(self.crosstrack_error) > 1.5 or math.fabs(self.phi_error) > 1.4:
            done = True
        if self.ep_steps == self.max_ep_steps:
            done = True
            self.ep_steps = 0
        self.reward = (
            (2.0 - math.fabs(self.crosstrack_error))
            * (4.5 - math.fabs(self.vel_error))
            * (math.pi / 3.0 - math.fabs(self.phi_error))
            - math.fabs(self.action[0] - self.prev_action[0])
            - 2 * math.fabs(self.action[1])
        )
        self.omega_reward = -2 * math.fabs(self.action[1])
        self.vel_reward = -math.fabs(self.action[0] - self.prev_action[0])
        self.prev_action = self.action
        if self.waypoints_list[k][3] >= 2.5 and math.fabs(self.vel_error) > 1.5:
            self.reward = 0
        elif self.waypoints_list[k][3] < 2.5 and math.fabs(self.vel_error) > 0.5:
            self.reward = 0
            
        self.total_ep_reward = self.total_ep_reward + self.reward
        return obs, self.reward, done, {}

    
    def reset(self):
        self.ep_start = 1
        self.ep_poses = []
        self.total_ep_reward = 0
        if self.max_vel >= 5:
            self.max_vel = 1
        index = np.random.randint(self.number_of_waypoints, size=1)[0]
        self.closest_index = index
        self.prev_closest_index = index
        self.pose[0] = self.waypoints_list[index][0] + 0.1
        self.pose[1] = self.waypoints_list[index][1] + 0.1
        self.pose[2] = self.waypoints_list[index][2] + 0.01
        self.xpose = [self.pose[0]] * self.n_traj
        self.ypose = [self.pose[1]] * self.n_traj
        self.twist = [0.0, 0.0, 0.0]
        for i in range(0, self.number_of_waypoints):
            if self.ref_vel[i] > self.max_vel:
                self.waypoints_list[i][3] = self.max_vel
            else:
                self.waypoints_list[i][3] = self.ref_vel[i]
        # self.max_vel = 2
        self.max_vel = self.max_vel + 1
        obs = self.get_observation()
        return obs
        
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
        self.prev_angle = self.pose[2]
        self.pose[0] = x + v_ * math.cos(th) * dt
        self.pose[1] = y + v_ * math.sin(th) * dt
        self.pose[2] = th + w_ * dt
        self.ep_poses.append(np.array([x, y, th, v_, w_, v, w]))
        self.ep_start = 0

    def get_observation(self):
        obs   = [0] * (self.horizon * 4 + 2)
        pose  = self.pose
        twist = self.twist
        index   = self.closest_index
        
        self.closest_dist = math.inf
        for i in range(self.closest_index, self.number_of_waypoints):
            dist = get_dist(self.waypoints_list[i], pose)
            if dist <= self.closest_dist:
                self.closest_dist = dist
                index = i
            else:
                break
        self.closest_index = index
        
        j = 0
        for i in range(0, self.horizon):
            k = i + self.closest_index
            if k < self.number_of_waypoints:
                r = get_dist(self.waypoints_list[k], pose)
                xdiff = self.waypoints_list[k][0] - pose[0]
                ydiff = self.waypoints_list[k][1] - pose[1]
                th = get_theta(xdiff, ydiff)
                vehicle_th = zero_to_2pi(pose[2])
                # vehicle_th = -vehicle_th
                # vehicle_th = 2*math.pi - vehicle_th
                yaw_error = pi_to_pi(self.waypoints_list[k][2] - vehicle_th)
                vel = self.waypoints_list[k][3]
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
        return obs

def read_waypoint_file(filename):
    num_waypoints = 0
    waypoints_list = []
    ref_vel = []
    with open(filename) as csv_file:
        pos = csv.reader(csv_file, delimiter=",")
        for index, row in enumerate(pos):
            if index == 0: # skip column names
                continue
            utm_cord = [float(row[0]), float(row[1])]
            phi = 0.0
            xcoord = utm_cord[0] * math.cos(phi) + utm_cord[1] * math.sin(phi)
            ycoord = -utm_cord[0] * math.sin(phi) + utm_cord[1] * math.cos(phi)
            waypoints_list.append(
                np.array([utm_cord[0], utm_cord[1], float(row[2]), float(row[3])])
            )
            ref_vel.append(float(row[3]))
        # waypoints_list.append(np.array([utm_cord[0], utm_cord[1], float(row[2]), 1.5]))
        for i in range(0, len(waypoints_list) - 1):
            xdiff = waypoints_list[i + 1][0] - waypoints_list[i][0]
            ydiff = waypoints_list[i + 1][1] - waypoints_list[i][1]
            waypoints_list[i][2] = zero_to_2pi(
                get_theta(xdiff, ydiff)
            )
        waypoints_list[i + 1][2] = waypoints_list[i][2]
        num_waypoints = i + 2
    
    return waypoints_list, ref_vel, num_waypoints

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

def get_dist(waypoint, pose):
    xdiff = pose[0] - waypoint[0]
    ydiff = pose[1] - waypoint[1]
    return math.sqrt(xdiff * xdiff + ydiff * ydiff)


def get_theta(xdiff, ydiff):
    theta = math.atan2(ydiff, xdiff)
    return zero_to_2pi(theta)



import ez_yaml

config = ez_yaml.to_object(string="""
vehicle:
    name: Warthog
    real_length: 1 # meter (approximately)
    real_width: 0.5 # meters (approximately)
    render_length: 0.25 
    render_width: 0.5
    controller_max_velocity: 4 # meters per second
    controller_max_spin: 2.5 # radians

reward_parameters:
    max_expected_crosstrack_error:  2.0   # meters
    max_expected_velocity_error:    1.125 # scaled by the vehicle's controller max velocity (1.1 = 110%)
    max_expected_angle_error:       1.047 # scaled by the vehicle's controller max angle, after scaling the units are radians
    velocity_jerk_cost_coefficient: 1
    spin_jerk_cost_coefficient:     0.5
    direct_velocity_cost:           0 # no penalty for going fast
    direct_spin_cost:               1 # no scaling of spin cost relative to combination
    
    # for reward_function2
    distance_scale: 20  # base distance is 0 to 1 (is 1 when directly on point, 0 when infinitely far away) then that is scaled by this value
                        # this number is important relative to the size of the reward from crosstrack/velocity
                        # bigger=distance is more important than crosstrack or angle
    completed_waypoint_bonus: 100 # this value is added for every completed waypoint (e.g. progress along the path is good; not just being close to the same point over and over)
    
    velocity_caps_enabled: true
    velocity_caps:
        # EXAMPLE:
        #   40%: 10% 
        #  # ^this means, if the closest waypoint has a velocity >= 40% of max-speed,
        #  # then the velocity error must be < 10% of max speed (otherwise 0 reward)
        
        0%:    12.5% # 0.5m/s for warthog is (0.5/4.0) => 0.125 => 12.5%
        62.5%: 37.5% # 2.5m/s for warthog is (2.5/4.0) => 0.625 => 62.5%, 1.5m/s is (1.5/4.0) => 37.5%
""")


def original_reward_function(*, spacial_info, closest_distance, relative_velocity, prev_relative_velocity, relative_spin, prev_relative_spin, closest_waypoint):
    x_diff     = closest_waypoint.x - spacial_info.x
    y_diff     = closest_waypoint.y - spacial_info.y
    angle_diff = get_theta(x_diff, y_diff)
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