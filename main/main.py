from envs.warthog import WarthogEnv
from action_adjuster import ActionAdjuster
from config import config, path_to
import torch
from rigorous_recorder import RecordKeeper
from stable_baselines3 import PPO
from blissful_basics import FS
from specific_tools.train_ppo import * # required because of pickle lookup

from statistics import mean as average
from random import random, sample, choices

recorder = RecordKeeper(config=config)


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
# action adjuster
# 
action_adjuster = ActionAdjuster(policy=policy, recorder=recorder)

# 
# env
# 
env = WarthogEnv(
    waypoint_file_path=path_to.default_waypoints,
    trajectory_output_path=f"{path_to.default_output_folder}/trajectory.log"
)

observation = env.reset()
accumulated_reward = 0
while True:
    action          = policy(observation)
    adjusted_action = action_adjuster.transform.modify_action(action)
    observation, reward, done, additional_info = env.step(adjusted_action)
    accumulated_reward += reward
    recorder.add(accumulated_reward=accumulated_reward)
    recorder.add(reward=reward)
    action_adjuster.add_data(observation, additional_info)
    if done:
        break