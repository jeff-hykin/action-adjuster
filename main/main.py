import os
from statistics import mean as average
from random import random, sample, choices
from copy import deepcopy

import torch
from stable_baselines3 import PPO

from __dependencies__.rigorous_recorder import RecordKeeper
from __dependencies__.blissful_basics import FS, print, LazyDict, run_main_hooks_if_needed

from specific_tools.train_ppo import * # required because of pickle lookup
from config import config, path_to, selected_profiles, grug_test
from data_structures import *
if config.use_simple:
    from envs.warthog_simple import WarthogEnv
else:
    from envs.warthog import WarthogEnv
from action_adjuster import ActionAdjustedAgent, NormalAgent
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
    waypoints_list=tuple(getattr(env, "waypoints_list", getattr(env,"_", {}).get("waypoints_list", None))),
)

# 
# Runtime
# 
if __name__ == '__main__':
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