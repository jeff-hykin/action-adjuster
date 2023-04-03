from dataclasses import dataclass, field
import json_fix
from super_map import LazyDict

@dataclass
class Timestep:
    index        : int   = None
    observation  : None  = None 
    reaction     : None  = None 
    reward       : float = None 
    is_last_step : bool  = False
    hidden_info  : None  = None
    
    def __init__(self, timestep=None, *, index=None, observation=None, reaction=None, reward=None, is_last_step=None, hidden_info=None, **kwargs):
        if timestep:
            for each_attr in dir(timestep):
                # skip magic attributes
                if len(each_attr) > 2 and each_attr[0:2] == '__':
                    continue
                # adopt all other attributes
                setattr(self, each_attr, getattr(timestep, each_attr))
        
        # set any non-None values
        self.index        = index        if not (type(index       ) == type(None)) else self.index
        self.observation  = observation  if not (type(observation ) == type(None)) else self.observation
        self.reaction     = reaction     if not (type(reaction    ) == type(None)) else self.reaction
        self.reward       = reward       if not (type(reward      ) == type(None)) else self.reward
        self.is_last_step = is_last_step if not (type(is_last_step) == type(None)) else self.is_last_step
        self.hidden_info  = hidden_info  if not (type(hidden_info ) == type(None)) else self.hidden_info
    
    def __json__(self):
        return self.__dict__
    
    def __repr__(self):
        return "Timestep"+repr(LazyDict(
            index=self.index,
            observation=self.observation,
            reaction=self.reaction,
            reward=self.reward,
            is_last_step=self.is_last_step,
            hidden_info=self.hidden_info,
        ))
    
    @classmethod
    def from_dict(cls, dict_data):
        return Timestep(**dict_data)
    
    def __hash__(self):
        return hash((
            self.index,
            self.observation,
            self.reaction,
            self.reward,
            self.is_last_step,
        ))

class MockTimestep(Timestep):
    def __init__(self, timestep, *, index):
        self.__dict__ = timestep.__dict__
        self._index = index
    
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, value):
        self._index = value
    

class TimestepSeries:
    def __init__(self, timesteps=None):
        self.index = -1
        self.steps = {}
        if type(timesteps) != type(None):
            for each in timesteps:
                self.index = each.index
                self.steps[self.index] = each
    
    @property
    def prev(self):
        if self.index > 0:
            return self.steps[self.index-1]
        else:
            return Timestep() # all attributes are none/false
    
    def add(self, state=None, reaction=None, reward=None, is_last_step=False, hidden_info=None):
        # if timestep, pull all the data out of the timestep
        if isinstance(state, Timestep):
            observation  = state.observation
            reaction     = state.reaction
            reward       = state.reward
            is_last_step = state.is_last_step
            hidden_info  = state.hidden_info
            
        self.index += 1
        self.steps[self.index] = Timestep(index=self.index, observation=observation, reaction=reaction, reward=reward, is_last_step=is_last_step, hidden_info=hidden_info)
    
    @property
    def observations(self):
        return [ each.observation for each in self.steps.values() ]
    
    @property
    def reactions(self):
        return [ each.reaction for each in self.steps.values() ]
    
    @property
    def rewards(self):
        return [ each.reward for each in self.steps.values() ]
    
    @property
    def is_last_steps(self):
        return [ each.reward for each in self.steps.values() ]
    
    def items(self):
        """
        for index, state, reaction, reward, next_state in time_series.items():
            pass
        """
        return ((each.index, each.observation, each.reaction, each.reward, each.is_last_step) for each in self.steps.values())
    
    def __len__(self):
        return len(self.steps)
        
    def __getitem__(self, key):
        if isinstance(key, float):
            key = int(key)
            
        if isinstance(key, int):
            if key < 0:
                key = self.index + key
                if key < 0:
                    return Timestep() # too far back
            # generate steps as needed
            while key > self.index:
                self.add()
            return self.steps[key]
        else:
            new_steps = { 
                each_key: Timestep(self.steps[each_key])
                    for each_key in range(key.start, key.stop, key.step) 
            }
            time_slice = TimestepSeries()
            time_slice.index = max(new_steps.keys())
            time_slice.steps = new_steps
            return time_slice
    
    def __repr__(self):
        string = "TimestepSeries(\n"
        for index, observation, reaction, reward, is_last_step in self.items():
            string += f"    {index}, {observation}, {reaction}, {reward}, {is_last_step}\n"
        string += ")"
        return string
    
    def __hash__(self):
        return hash(tuple(self.steps.values()))

    def __eq__(self, other):
        return hash(self) == hash(other)