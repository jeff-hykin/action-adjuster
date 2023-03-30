from stable_baselines3 import PPO
from blissful_basics import FS

from statistics import mean as average
from random import random, sample, choices

agent = PPO.load(path_to.saved_policies)

hidden_state = None
def policy(observation):
    global hidden_state
    action, hidden_state = agent.predict(observation, hidden_state)
    
    # predict(
    #     observation: Union[numpy.ndarray, Dict[str, numpy.ndarray]],
    #     state: Optional[Tuple[numpy.ndarray, ...]] = None,
    #     episode_start: Optional[numpy.ndarray] = None,
    #     deterministic: bool = False
    # )
    # return 
    #     Tuple[numpy.ndarray, Optional[Tuple[numpy.ndarray, ...]]]
    # 
    #     Get the policy action from an observation (and optional hidden state).
    #     Includes sugar-coating to handle different observations (e.g. normalizing images).
        
    #     :param observation: the input observation
    #     :param state: The last hidden states (can be None, used in recurrent policies)
    #     :param episode_start: The last masks (can be None, used in recurrent policies)
    #         this correspond to beginning of episodes,
    #         where the hidden states of the RNN must be reset.
    #     :param deterministic: Whether or not to return deterministic actions.
    #     :return: the model's action and the next hidden state
    #         (used in recurrent policies)
    
    return action
    
