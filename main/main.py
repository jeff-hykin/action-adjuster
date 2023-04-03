from statistics import mean as average
from random import random, sample, choices

import torch
from rigorous_recorder import RecordKeeper
from stable_baselines3 import PPO
from blissful_basics import FS, print, LazyDict

from envs.warthog import WarthogEnv
from action_adjuster import ActionAdjustedAgent
from config import config, path_to
from specific_tools.train_ppo import * # required because of pickle lookup
from generic_tools.universe.agent import Skeleton
from generic_tools.universe.timestep import Timestep
import generic_tools.universe.runtimes as runtimes
from runtimes.warthog_ros import RosRuntime

recorder = RecordKeeper(config=config)

# 
# env
# 
env = WarthogEnv(
    waypoint_file_path=path_to.default_waypoints,
    trajectory_output_path=f"{path_to.default_output_folder}/trajectory.log",
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
agent = ActionAdjustedAgent(
    observation_space=env.observation_space,
    reaction_space=env.action_space,
    policy=policy,
    recorder=recorder,
)

# 
# Runtime
# 
if config.should_use_ros:
    RosRuntime(agent=agent, env=env)
else:
    # basic runtime
    for episode_index, timestep_index, observation, reward, is_last_step in runtimes.basic(agent=agent, env=env):
        pass