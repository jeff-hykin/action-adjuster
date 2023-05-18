from stable_baselines3 import PPO
from blissful_basics import FS
from config import config, path_to, absolute_path_to

from statistics import mean as average
from random import random, sample, choices

agent = PPO.load(path_to.saved_policies+"/bicycle1")
# agent = PPO.load("../.ignore.repo/warthog_rl/policy/parallel")

hidden_state = None
def policy(observation):
    global hidden_state
    action, hidden_state = agent.predict(observation, hidden_state, deterministic=True)
    
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
    

# .ignore.repo/warthog_rl/policy/bicycle1.zip
# .ignore.repo/warthog_rl/policy/bicycle2.zip
# .ignore.repo/warthog_rl/policy/bicycle3.zip
# .ignore.repo/warthog_rl/policy/bicycle4.zip
# .ignore.repo/warthog_rl/policy/model_continuous.zip
# .ignore.repo/warthog_rl/policy/model_final.zip
# .ignore.repo/warthog_rl/policy/model1.zip
# .ignore.repo/warthog_rl/policy/model2.zip
# .ignore.repo/warthog_rl/policy/parallel.zip
# .ignore.repo/warthog_rl/policy/parallel2.zip
# .ignore.repo/warthog_rl/policy/vel_dweight0.zip
# .ignore.repo/warthog_rl/policy/vel_dweight1.zip
# .ignore.repo/warthog_rl/policy/vel_dweight2.zip
# .ignore.repo/warthog_rl/policy/vel_dweight3.zip
# .ignore.repo/warthog_rl/policy/vel_dweight4.zip
# .ignore.repo/warthog_rl/policy/vel_dweight5.zip
# .ignore.repo/warthog_rl/policy/vel_dweight6.zip
# .ignore.repo/warthog_rl/policy/vel_dweight7.zip
# .ignore.repo/warthog_rl/policy/vel_dweight8.zip
# .ignore.repo/warthog_rl/policy/vel_dweight9.zip
# .ignore.repo/warthog_rl/policy/vel_weight0.zip
# .ignore.repo/warthog_rl/policy/vel_weight1.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d0.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d1.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d2.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d3.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d4.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d5.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d6.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d7.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d8.zip
# .ignore.repo/warthog_rl/policy/vel_weight2_d9.zip
# .ignore.repo/warthog_rl/policy/vel_weight2.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d0.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d1.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d2.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d3.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d4.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d5.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d6.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d7.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d8.zip
# .ignore.repo/warthog_rl/policy/vel_weight3_d9.zip
# .ignore.repo/warthog_rl/policy/vel_weight3.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d0.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d1.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d2.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d3.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d4.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d5.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d6.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d7.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d8.zip
# .ignore.repo/warthog_rl/policy/vel_weight4_d9.zip
# .ignore.repo/warthog_rl/policy/vel_weight4.zip
# .ignore.repo/warthog_rl/policy/vel_weight5_d0.zip
# .ignore.repo/warthog_rl/policy/vel_weight5.zip
# .ignore.repo/warthog_rl/policy/vel_weight6.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d0.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d1.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d2.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d3.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d4.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d5.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d6.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d7.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d8.zip
# .ignore.repo/warthog_rl/policy/vel_weight7_d9.zip
# .ignore.repo/warthog_rl/policy/vel_weight7.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d0.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d1.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d10.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d11.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d12.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d2.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d3.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d4.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d5.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d6.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d7.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d8.zip
# .ignore.repo/warthog_rl/policy/vel_weight8_d9.zip
# .ignore.repo/warthog_rl/policy/vel_weight8.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d0.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d1.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d2.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d3.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d4.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d5.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d6.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d7.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d8.zip
# .ignore.repo/warthog_rl/policy/vel_weight9_d9.zip
# .ignore.repo/warthog_rl/policy/vel_weight9.zip