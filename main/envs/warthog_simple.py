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
from config import grug_test, path_to

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


@grug_test
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


# 
# data structures (for test cases)
# 
if True:
    StepOutput = namedtuple('StepOutput', ['observation', 'reward', 'done', 'debug'])
    StepSideEffects = namedtuple('StepSideEffects', [
        'action',
        'crosstrack_error',
        'ep_steps',
        'num_steps',
        'omega_reward',
        'phi_error',
        'prev_action',
        'prev_closest_index',
        'reward',
        'total_ep_reward',
        'vel_error',
        'vel_reward',
        'twist',
        'prev_angle',
        'pose',
        'ep_start',
        'closest_dist',
        'closest_index',
    ])
    GetObservationOutput = namedtuple('GetObservationOutput', [
        'obs',
        'closest_dist',
        'closest_index'
    ])
    WarthogSimOutput = namedtuple('WarthogSimOutput', [
        'twist',
        'prev_angle',
        'pose',
        'ep_start',
    ])


@grug_test(max_io=30)
def pure_sim_warthog(
    v, 
    w, 
    pose,
    twist,
    is_delayed_dynamics,
    v_delay_data,
    w_delay_data,
    dt,
    prev_angle,
    ep_poses,
    ep_start,
):
    x = pose[0]
    y = pose[1]
    th = pose[2]
    v_ = twist[0]
    w_ = twist[1]
    twist[0] = v
    twist[1] = w
    if is_delayed_dynamics:
        v_ = v_delay_data[0]
        w_ = w_delay_data[0]
        del v_delay_data[0]
        del w_delay_data[0]
        v_delay_data.append(v)
        w_delay_data.append(w)
        twist[0] = v_delay_data[0]
        twist[1] = v_delay_data[1]
    dt = dt
    prev_angle = pose[2]
    pose[0] = x + v_ * math.cos(th) * dt
    pose[1] = y + v_ * math.sin(th) * dt
    pose[2] = th + w_ * dt
    ep_poses.append(np.array([x, y, th, v_, w_, v, w]))
    ep_start = 0
    
    return twist, prev_angle, pose, ep_start

@grug_test(max_io=30)
def pure_get_observation(
    closest_dist,
    closest_index,
    horizon,
    number_of_waypoints,
    pose,
    twist,
    waypoints_list,
):
    obs   = [0] * (horizon * 4 + 2)
    index   = closest_index
    
    closest_dist = math.inf
    for i in range(closest_index, number_of_waypoints):
        dist = get_dist(waypoints_list[i], pose)
        if dist <= closest_dist:
            closest_dist = dist
            index = i
        else:
            break
    closest_index = index
    
    j = 0
    for i in range(0, horizon):
        k = i + closest_index
        if k < number_of_waypoints:
            r = get_dist(waypoints_list[k], pose)
            xdiff = waypoints_list[k][0] - pose[0]
            ydiff = waypoints_list[k][1] - pose[1]
            th = get_theta(xdiff, ydiff)
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
    
    return GetObservationOutput(obs, closest_dist, closest_index)

@grug_test(max_io=30)
def pure_step(
    ep_steps,
    num_steps,
    action,
    prev_closest_index,
    closest_index,
    number_of_waypoints,
    waypoints_list,
    pose,
    phi_error,
    vel_error,
    twist,
    crosstrack_error,
    closest_dist,
    max_ep_steps,
    reward,
    prev_action,
    omega_reward,
    vel_reward,
    total_ep_reward,
    is_delayed_dynamics,
    v_delay_data,
    w_delay_data,
    dt,
    prev_angle,
    ep_poses,
    ep_start,
    horizon,
):
    ep_steps = ep_steps + 1
    num_steps = num_steps + 1
    action[0] = np.clip(action[0], 0, 1) * 4.0
    action[1] = np.clip(action[1], -1, 1) * 2.5
    action = action
    twist, prev_angle, pose, ep_start = pure_sim_warthog(
        v=action[0],
        w=action[1],
        pose=pose,
        twist=twist,
        is_delayed_dynamics=is_delayed_dynamics,
        v_delay_data=v_delay_data,
        w_delay_data=w_delay_data,
        dt=dt,
        prev_angle=prev_angle,
        ep_poses=ep_poses,
        ep_start=ep_start,
    )
    prev_closest_index = closest_index
    obs, closest_dist, closest_index = pure_get_observation(
        closest_dist=closest_dist,
        closest_index=closest_index,
        horizon=horizon,
        number_of_waypoints=number_of_waypoints,
        pose=pose,
        twist=twist,
        waypoints_list=waypoints_list,
    )
    done = False
    if closest_index >= number_of_waypoints - 1:
        done = True
    
    # 
    # Calculating reward
    # 
    k = closest_index
    xdiff = waypoints_list[k][0] - pose[0]
    ydiff = waypoints_list[k][1] - pose[1]
    th = get_theta(xdiff, ydiff)
    yaw_error = pi_to_pi(th - pose[2])
    phi_error = pi_to_pi(
        waypoints_list[closest_index][2] - pose[2]
    )
    vel_error = waypoints_list[k][3] - twist[0]
    crosstrack_error = closest_dist * math.sin(yaw_error)
    if math.fabs(crosstrack_error) > 1.5 or math.fabs(phi_error) > 1.4:
        done = True
    if ep_steps == max_ep_steps:
        done = True
        ep_steps = 0
    reward = (
        (2.0 - math.fabs(crosstrack_error))
        * (4.5 - math.fabs(vel_error))
        * (math.pi / 3.0 - math.fabs(phi_error))
        - math.fabs(action[0] - prev_action[0])
        - 2 * math.fabs(action[1])
    )
    omega_reward = -2 * math.fabs(action[1])
    vel_reward = -math.fabs(action[0] - prev_action[0])
    prev_action = action
    if waypoints_list[k][3] >= 2.5 and math.fabs(vel_error) > 1.5:
        reward = 0
    elif waypoints_list[k][3] < 2.5 and math.fabs(vel_error) > 0.5:
        reward = 0
        
    total_ep_reward = total_ep_reward + reward
    return (
        StepOutput(
            obs,
            reward,
            done,
            {}
        ),
        StepSideEffects(
            action,
            crosstrack_error,
            ep_steps,
            num_steps,
            omega_reward,
            phi_error,
            prev_action,
            prev_closest_index,
            reward,
            total_ep_reward,
            vel_error,
            vel_reward,
            twist,
            prev_angle,
            pose,
            ep_start,
            closest_dist,
            closest_index,
        ),
        (
            yaw_error,
            ydiff,
            xdiff,
        )
    )

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
    
    # just a wrapper around the pure_step
    def step(self, action):
        output, (self.action, self.crosstrack_error, self.ep_steps, self.num_steps, self.omega_reward, self.phi_error, self.prev_action, self.prev_closest_index, self.reward, self.total_ep_reward, self.vel_error, self.vel_reward, self.twist, self.prev_angle, self.pose, self.ep_start, self.closest_dist, self.closest_index,), other = pure_step(
            ep_steps=self.ep_steps,
            num_steps=self.num_steps,
            action=self.action,
            prev_closest_index=self.prev_closest_index,
            closest_index=self.closest_index,
            number_of_waypoints=self.number_of_waypoints,
            waypoints_list=self.waypoints_list,
            pose=self.pose,
            phi_error=self.phi_error,
            vel_error=self.vel_error,
            twist=self.twist,
            crosstrack_error=self.crosstrack_error,
            closest_dist=self.closest_dist,
            max_ep_steps=self.max_ep_steps,
            reward=self.reward,
            prev_action=self.prev_action,
            omega_reward=self.omega_reward,
            vel_reward=self.vel_reward,
            total_ep_reward=self.total_ep_reward,
            is_delayed_dynamics=self.is_delayed_dynamics,
            v_delay_data=self.v_delay_data,
            w_delay_data=self.w_delay_data,
            dt=self.dt,
            prev_angle=self.prev_angle,
            ep_poses=self.ep_poses,
            ep_start=self.ep_start,
            horizon=self.horizon,
        )
        return output
    
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
        obs, self.closest_dist, self.closest_index = pure_get_observation(
            closest_dist=self.closest_dist,
            closest_index=self.closest_index,
            horizon=self.horizon,
            number_of_waypoints=self.number_of_waypoints,
            pose=self.pose,
            twist=self.twist,
            waypoints_list=self.waypoints_list,
        )
        return obs

if not grug_test.fully_disable and (grug_test.replay_inputs or grug_test.record_io):
    @grug_test
    def smoke_test_warthog(trajectory_file):
        env = WarthogEnv(path_to.waypoints_folder+f"/{trajectory_file}")
        outputs = []
        def env_snapshot(env):
            return deepcopy(dict(
                waypoints_list=env.waypoints_list,
                pose=env.pose,
                twist=env.twist,
                closest_index=env.closest_index,
                prev_closest_index=env.prev_closest_index,
                closest_dist=env.closest_dist,
                number_of_waypoints=env.number_of_waypoints,
                horizon=env.horizon,
                dt=env.dt,
                ref_vel=env.ref_vel,
                num_steps=env.num_steps,
                max_vel=env.max_vel,
                waypoints_dist=env.waypoints_dist,
                warthog_length=env.warthog_length,
                warthog_width=env.warthog_width,
                warthog_diag=env.warthog_diag,
                diag_angle=env.diag_angle,
                prev_angle=env.prev_angle,
                n_traj=env.n_traj,
                xpose=env.xpose,
                ypose=env.ypose,
                crosstrack_error=env.crosstrack_error,
                vel_error=env.vel_error,
                phi_error=env.phi_error,
                start_step_for_sup_data=env.start_step_for_sup_data,
                ep_steps=env.ep_steps,
                max_ep_steps=env.max_ep_steps,
                tprev=env.tprev,
                total_ep_reward=env.total_ep_reward,
                reward=env.reward,
                action=env.action,
                prev_action=env.prev_action,
                omega_reward=env.omega_reward,
                vel_reward=env.vel_reward,
                is_delayed_dynamics=env.is_delayed_dynamics,
                delay_steps=env.delay_steps,
                v_delay_data=env.v_delay_data,
                w_delay_data=env.w_delay_data,
                save_data=env.save_data,
                ep_start=env.ep_start,
                ep_dist=env.ep_dist,
                ep_poses=env.ep_poses,
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
    
    smoke_test_warthog("real1.csv")

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