from envs.warthog import WarthogEnv
from action_adjuster import ActionAdjuster
from config import config, path_to
import torch
from rigorous_recorder import RecordKeeper

from statistics import mean as average
from random import random, sample, choices

recorder = RecordKeeper(config=config)

# FIXME: need a real policy
def policy(observation):
    velocity_action = 0.5
    spin_action     = 0.00
    return velocity_action, spin_action

action_adjuster = ActionAdjuster(policy=policy, recorder=recorder)

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