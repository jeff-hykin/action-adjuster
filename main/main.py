import os
from statistics import mean as average
from random import random, sample, choices
from copy import deepcopy

import torch
from stable_baselines3 import PPO

from __dependencies__.rigorous_recorder import RecordKeeper
from __dependencies__.blissful_basics import FS, print, LazyDict, run_main_hooks_if_needed

from specific_tools.train_ppo import * # required because of pickle lookup
# from envs.warthog import WarthogEnv
from envs.warthog_simple import WarthogEnv
from action_adjuster import ActionAdjustedAgent, NormalAgent
from config import config, path_to, selected_profiles, grug_test
from generic_tools.functions import cache_outputs
from generic_tools.universe.agent import Skeleton
from generic_tools.universe.timestep import Timestep
import generic_tools.universe.runtimes as runtimes

# 
# policy
# 
if config.policy.name == 'dummy':
    from policies.dummy import policy
if config.policy.name == 'bicycle':
    from policies.bicycle import policy
if config.policy.name == 'kinematic_sup0':
    from policies.kinematic_sup0 import policy
if config.policy.name == 'retrained':
    from policies.retrained import policy
if config.policy.name == 'original':
    from policies.original import policy

run_main_hooks_if_needed(__name__)

recorder = RecordKeeper(selected_profiles=selected_profiles, config=config)

# 
# env
# 
env = WarthogEnv(
    path_to.default_waypoints,
    trajectory_output_path=f"{config.output_folder}/trajectory.log",
    recorder=recorder,
)

# this is just a means of making the policy pseudo-deterministic, not intentionally an optimization
# TODO: if anything this will act like a memory leak, so it needs to be capped. However, to not break things its cap size depends on how quickly the fit_points is called and how big the buffer is
#       which is why its not capped right now
policy = cache_outputs(policy)


# 
# agent
# 
Agent = NormalAgent if config.action_adjuster.disabled else ActionAdjustedAgent
agent = Agent(
    observation_space=env.observation_space,
    reaction_space=env.action_space,
    policy=policy,
    recorder=recorder,
    waypoints_list=tuple(env.waypoints_list),
)

# 
# Runtime
# 
if __name__ == '__main__':
    if not grug_test.fully_disable and (grug_test.replay_inputs or grug_test.record_io):
        @grug_test(max_io=2)
        def main_smoke_test_warthog(trajectory_file):
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
                        next_waypoint_index_=                          getattr(env._, "next_waypoint_index_"                                 , None),
                        prev_next_waypoint_index_=                     getattr(env._, "prev_next_waypoint_index_"                            , None),
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
                        relative_action=                               getattr(env._, "relative_action"                               , None),
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
        
        main_smoke_test_warthog("real1.csv")
        main_smoke_test_warthog("concrete1.csv")
        if grug_test.replay_inputs: exit()
    
    with print.indent:
        if config.should_use_ros:
            from runtimes.warthog_ros_client import RosRuntime, rospy
            rospy.init_node(config.ros_runtime.main_node_name)
            RosRuntime(agent=agent, env=env)
            rospy.spin()
        else:
            # basic runtime
            for episode_index, timestep_index, observation, reward, is_last_step in runtimes.basic(agent=agent, env=env  , max_timestep_index=config.simulator.max_number_of_timesteps_per_episode):
                pass
            
            print("done")
            import subprocess
            # for some multi-threading reason this process doesn't close after exit
            pid = os.getpid()
            print(f"called kill on self {os.getpid()}, now exiting")
            stdout = subprocess.check_output(['kill', f"{pid}"]).decode('utf-8')[0:-1]
            
            from time import sleep
            sleep(1)
            print(f"calling kill -9 on self {pid}")
            stdout = subprocess.check_output(['kill', '-9', f"{pid}"]).decode('utf-8')[0:-1]