from statistics import mean as average
from random import random, sample, choices

import torch
from blissful_basics import FS, print, stringify

from __dependencies__.super_hash import super_hash
from trivial_torch_tools.misc import DeterministicTorchRng
from config import config, path_to, absolute_path_to
from specific_tools.train_ppo import PolicyNetworkGauss, device

pi = torch.load(path_to.saved_policies+ "/manaul_ppo_10000000.pt")

def policy(observation):
    with print.indent:
        observation = torch.as_tensor(observation, dtype=torch.float32).to(device)
        distribution = pi(observation)
        with DeterministicTorchRng(observation):
            action = distribution.sample()
        
        return action.cpu().numpy()