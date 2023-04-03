import itertools
import math
from copy import deepcopy
from generic_tools.universe.timestep import Timestep


def basic(*, agent, env, max_timestep_index=math.inf, max_episode_index=math.inf, observation_modifier=None, reaction_modifier=None, reward_modifier=None):
    """
    for episode_index, timestep_index, agent.timestep.observation, agent.timestep.reward, agent.timestep.is_last_step in traditional_runtime(agent=agent, env=env):
        pass
    """
    if observation_modifier is None: observation_modifier = lambda each: each
    if reaction_modifier    is None: reaction_modifier    = lambda each: each
    if reward_modifier      is None: reward_modifier      = lambda each: each
    
    agent.when_mission_starts()
    timestep_count = 0
    for episode_index in itertools.count(0): # starting at 0
        
        agent.previous_timestep = Timestep(
            index=-2,
        )
        agent.timestep = Timestep(
            index=-1,
        )
        agent.next_timestep = Timestep(
            index=0,
            observation=observation_modifier(deepcopy(env.reset())),
            is_last_step=False,
        )
        agent.when_episode_starts()
        while not agent.timestep.is_last_step:
            timestep_count += 1
            if timestep_count >= max_timestep_index:
                break
            
            agent.previous_timestep = agent.timestep
            agent.timestep          = agent.next_timestep
            agent.next_timestep     = Timestep(index=agent.next_timestep.index+1)
            
            agent.when_timestep_starts()
            reaction = agent.timestep.reaction
            if type(reaction) == type(None):
                reaction = env.action_space.sample()
            reaction = reaction_modifier(reaction)
            observation, reward, is_last_step, agent.timestep.hidden_info = env.step(reaction)
            agent.next_timestep.observation = observation_modifier(deepcopy(observation))
            agent.timestep.reward           = reward_modifier(deepcopy(reward))
            agent.timestep.is_last_step     = deepcopy(is_last_step)
            agent.when_timestep_ends()
            
            yield episode_index, agent.timestep
        
        if timestep_count >= max_timestep_index:
            break
        agent.when_episode_ends()
    agent.when_mission_ends()