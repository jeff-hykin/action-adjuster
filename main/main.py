from statistics import mean as average
from random import random, sample, choices

import torch
from rigorous_recorder import RecordKeeper
from stable_baselines3 import PPO
from blissful_basics import FS, print, LazyDict

from envs.warthog import WarthogEnv
from action_adjuster import ActionAdjustedAgent, ActionAdjuster
from config import config, path_to, selected_profiles
from specific_tools.train_ppo import * # required because of pickle lookup
from generic_tools.universe.agent import Skeleton
from generic_tools.universe.timestep import Timestep
import generic_tools.universe.runtimes as runtimes

recorder = RecordKeeper(selected_profiles=selected_profiles, config=config)

# 
# env
# 
env = WarthogEnv(
    waypoint_file_path=path_to.default_waypoints,
    trajectory_output_path=f"{config.output_folder}/trajectory.log",
    recorder=recorder,
)

# 
# policy
# 
if config.policy.name == 'dummy':
    from policies.dummy import policy
if config.policy.name == 'bicycle':
    from policies.bicycle import policy
if config.policy.name == 'retrained':
    from policies.retrained import policy

# 
# agent
# 
ActionAdjuster.waypoints_list = env.waypoints_list # Hacky but needed so that waypoint data doesn't get sent across threads at every timesteps
agent = ActionAdjustedAgent(
    observation_space=env.observation_space,
    reaction_space=env.action_space,
    policy=policy,
    recorder=recorder,
)

# 
# Runtime
# 
with print.indent:
    if config.should_use_ros:
        from runtimes.warthog_ros_client import RosRuntime
        RosRuntime(agent=agent, env=env)
    else:
        # basic runtime
        for episode_index, timestep_index, observation, reward, is_last_step in runtimes.basic(agent=agent, env=env, max_timestep_index=699):
            pass
        
        print("done")
        exit() # exiting to kill other processes