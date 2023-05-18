from statistics import mean as average
from random import random, sample, choices

import torch
from blissful_basics import FS, print, stringify

from __dependencies__.super_hash import super_hash
from __dependencies__.trivial_torch_tools.misc import DeterministicTorchRng
from config import config, path_to, absolute_path_to
from specific_tools.train_ppo import PolicyNetworkGauss, device

pi = torch.load(path_to.saved_policies+ "/manaul_ppo_10000000.pt")

record_of_outputs = {}
def policy(observation):
    from envs.warthog import WarthogEnv
    with print.indent:
        original_observation = observation
        key = repr(original_observation)
        observation = torch.as_tensor(observation.to_numpy(), dtype=torch.float32).to(device)
        distribution = pi(observation)
        with DeterministicTorchRng(key):
            action = distribution.sample()
        
        action = action.cpu().numpy()
        
        if key in record_of_outputs:
            return record_of_outputs[key][0:2]
        
        record_of_outputs[key] = WarthogEnv.ReactionClass(action.tolist()+[original_observation.timestep])
        return action