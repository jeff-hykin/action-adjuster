import os
from statistics import mean as average
from random import random, sample, choices

import torch
from stable_baselines3 import PPO

from rigorous_recorder import RecordKeeper
from blissful_basics import FS, print, LazyDict, run_main_hooks_if_needed

from specific_tools.train_ppo import * # required because of pickle lookup
from envs.warthog import WarthogEnv
from action_adjuster import ActionAdjustedAgent
from config import config, path_to, selected_profiles
from generic_tools.functions import cache_outputs
from generic_tools.universe.agent import Skeleton
from generic_tools.universe.timestep import Timestep
import generic_tools.universe.runtimes as runtimes

run_main_hooks_if_needed(__name__)

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

# this is just a means of making the policy pseudo-deterministic, not intentionally an optimization
# TODO: if anything this will act like a memory leak, so it needs to be capped. However, to not break things its cap size depends on how quickly the fit_points is called and how big the buffer is
#       which is why its not capped right now
policy = cache_outputs(policy)


# 
# agent
# 
agent = ActionAdjustedAgent(
    observation_space=env.observation_space,
    reaction_space=env.action_space,
    policy=policy,
    recorder=recorder,
    waypoints_list=tuple(env.waypoints_list),
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
        import subprocess
        # for some multi-threading reason this process doesn't close after exit
        stdout = subprocess.check_output(['kill', '-9', f"{os.getpid()}"]).decode('utf-8')[0:-1]
        print("called kill -9 on self, now exiting")
        exit()