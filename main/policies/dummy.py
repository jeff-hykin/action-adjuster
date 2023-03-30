from config import config, path_to
from rigorous_recorder import RecordKeeper
from stable_baselines3 import PPO

from statistics import mean as average
from random import random, sample, choices

def policy(observation):
    velocity_action = 0.5
    spin_action     = 0.00
    return velocity_action, spin_action