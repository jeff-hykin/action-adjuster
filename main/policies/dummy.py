from config import config, path_to
from rigorous_recorder import RecordKeeper
from stable_baselines3 import PPO

from statistics import mean as average
from random import random, sample, choices

def policy(observation):
    velocity_action = config.dummy_policy.relative_velocity
    spin_action     = config.dummy_policy.relative_spin
    return velocity_action, spin_action