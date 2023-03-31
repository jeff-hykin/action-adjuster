from statistics import mean as average
from random import random, sample, choices

import torch
from blissful_basics import FS

from config import config, path_to, absolute_path_to
from specific_tools.train_ppo import PolicyNetworkGauss, device

pi = torch.load(path_to.saved_policies+ "/manaul_ppo_10000000.pt")

hidden_state = None
def policy(observation):
    distribution = pi(torch.as_tensor(observation, dtype=torch.float32).to(device))
    action = distribution.sample()
    return action.cpu().numpy()