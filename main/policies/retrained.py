from statistics import mean as average
from random import random, sample, choices

import torch

from __dependencies__.blissful_basics import FS, print, stringify, run_in_main
from __dependencies__.super_hash import super_hash
from __dependencies__.trivial_torch_tools.misc import DeterministicTorchRng

from config import config, path_to, absolute_path_to, debug, grug_test
from specific_tools.train_ppo import PolicyNetworkGauss, device

pi = None # load policy dynamically because the main thread needs to 
deterministic = True

@grug_test(max_io=100, record_io=None, additional_io_per_run=None)
def policy(observation):
    global pi
    pi = pi or torch.load(path_to.saved_policies+ "/manaul_ppo_10000000.pt")
    with print.indent:
        if hasattr(observation, "to_numpy"):
            observation = observation.to_numpy()
        observation = torch.as_tensor(observation, dtype=torch.float32).to(device)
        distribution = pi(observation)
        if deterministic:
            action = distribution.loc
        else:
            action = distribution.sample()
        return action.cpu().detach().numpy()


# this is an annoying but required workaround so that pickle can find the "PolicyNetworkGauss"
@run_in_main
def _():
    from specific_tools.train_ppo import PolicyNetworkGauss, device