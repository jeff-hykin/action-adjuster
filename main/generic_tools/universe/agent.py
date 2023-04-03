from generic_tools.universe.timestep import Timestep

class Skeleton:
    # TODO: add a "dont_show_help" and default to listing out all the attributes that enhancements give an agent
    previous_timestep = None
    timestep          = None
    next_timestep     = None
    # self.timestep.index
    # self.timestep.observation
    # self.timestep.is_last_step
    # self.timestep.reward
    # self.timestep.hidden_info
    # self.timestep.reaction
    
    def __init__(self, observation_space, reaction_space, **config):
        self.observation_space = observation_space
        self.reaction_space = reaction_space

    def when_mission_starts(self):
        pass
    def when_episode_starts(self):
        pass
    def when_timestep_starts(self):
        """
        read: self.timestep.observation
        write: self.timestep.reaction = something
        """
        pass
    def when_timestep_ends(self):
        """
        read: self.timestep.reward
        """
        pass
    def when_episode_ends(self):
        pass
    def when_mission_ends(self):
        pass

def enhance_with_single(enhancement_class):
    def wrapper1(init_function):
        def wrapper2(self, *args, **kwargs):
            real_init = enhancement_class
            def when_init(*args, **kwargs):
                return enhancement_class.when_init(self, real_init, *args, **kwargs)
            self.when_init = when_init
            
            real_mission_starts = self.when_mission_starts
            def when_mission_starts(*args, **kwargs):
                return enhancement_class.when_mission_starts(self, real_mission_starts, *args, **kwargs)
            self.when_mission_starts = when_mission_starts
            
            real_episode_starts = self.when_episode_starts
            def when_episode_starts(*args, **kwargs):
                return enhancement_class.when_episode_starts(self, real_episode_starts, *args, **kwargs)
            self.when_episode_starts = when_episode_starts
            
            real_timestep_starts = self.when_timestep_starts
            def when_timestep_starts(*args, **kwargs):
                return enhancement_class.when_timestep_starts(self, real_timestep_starts, *args, **kwargs)
            self.when_timestep_starts = when_timestep_starts
            
            real_timestep_ends = self.when_timestep_ends
            def when_timestep_ends(*args, **kwargs):
                return enhancement_class.when_timestep_ends(self, real_timestep_ends, *args, **kwargs)
            self.when_timestep_ends = when_timestep_ends
            
            real_episode_ends = self.when_episode_ends
            def when_episode_ends(*args, **kwargs):
                return enhancement_class.when_episode_ends(self, real_episode_ends, *args, **kwargs)
            self.when_episode_ends = when_episode_ends
            
            real_mission_ends = self.when_mission_ends
            def when_mission_ends(*args, **kwargs):
                return enhancement_class.when_mission_ends(self, real_mission_ends, *args, **kwargs)
            self.when_mission_ends = when_mission_ends
            
            has_custom_init = enhancement_class.__init__ != Enhancement.__init__
            if has_custom_init:
                enhancement_class.__init__(self, init_function, *args, **kwargs)
            else:
                init_function(self, *args, **kwargs)
        
        return wrapper2
    return wrapper1

def enhance_with(*enhancements):
    def wrapper(function_getting_wrapped):
        for each_enhancement in enhancements:
            function_getting_wrapped = enhance_with_single(each_enhancement)(function_getting_wrapped)
        return function_getting_wrapped
    return wrapper
        
class Enhancement:
    # TODO: attach all methods that are not these methods
    def __init__(self, *args, **kwargs):
        pass
    def when_mission_starts(self, original,):
        return original()
    def when_episode_starts(self, original,):
        return original()
    def when_timestep_starts(self, original,):
        return original()
    def when_timestep_ends(self, original,):
        return original()
    def when_episode_ends(self, original,):
        return original()
    def when_mission_ends(self, original,):
        return original()
