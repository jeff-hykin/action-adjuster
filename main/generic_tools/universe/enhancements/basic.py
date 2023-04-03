from super_map import LazyDict
from tools.basics import sort_keys, randomly_pick_from
from tools.universe.agent import Enhancement
from tools.universe.timestep import TimestepSeries, Timestep, MockTimestep

class EpisodeEnhancement(Enhancement):
    """
        creates:
            self.episode
            self.episode.index
            self.episode.reward
            self.episode.timestep
            self.episode.previous_timestep
            self.episodes[0]
    """
    
    def when_mission_starts(self, original):
        self.episode = LazyDict(
            index=-1,
            previous_timestep=MockTimestep(
                Timestep(),
                index=-2,
            ),
            timestep=MockTimestep(
                Timestep(),
                index=-1
            ),
            next_timestep=MockTimestep(
                Timestep(),
                index=0
            ),
        )
        self.episodes = []
        self.per_episode = LazyDict(
            average=LazyDict(
                reward=0,
            ),
        )
        
        original()
        
    def when_episode_starts(self, original):
        self.episode = LazyDict(
            index=self.episode.index+1,
            reward=0,
            previous_timestep=MockTimestep(
                self.previous_timestep,
                index=-2,
            ),
            timestep=MockTimestep(
                self.timestep,
                index=-1,
            ),
            next_timestep=MockTimestep(
                self.next_timestep,
                index=0
            ),
        )
        self.episodes.append(self.episode)
        original()
        
    
    def when_timestep_starts(self, original):
        self.episode.previous_timestep=MockTimestep(
            self.previous_timestep,
            index=self.episode.previous_timestep.index+1,
        )
        self.episode.timestep=MockTimestep(
            self.timestep,
            index=self.episode.timestep.index+1,
        )
        self.episode.next_timestep=MockTimestep(
            self.next_timestep,
            index=self.episode.next_timestep.index+1,
        )
        # get a reaction
        original()
    
    def when_timestep_ends(self, original):
        self.episode.previous_timestep=MockTimestep(
            self.previous_timestep,
            index=self.episode.previous_timestep.index,
        )
        self.episode.timestep=MockTimestep(
            self.timestep,
            index=self.episode.timestep.index,
        )
        self.episode.next_timestep=MockTimestep(
            self.next_timestep,
            index=self.episode.next_timestep.index,
        )
        # 
        # reward
        # 
        self.episode.reward += self.timestep.reward
        
        original()

class TimelineEnhancement(Enhancement):
    """
        needs:
            EpisodeEnhancement
        creates:
            self.timeline
    """
    
    def when_episode_starts(self, original, ):
        self.timeline = TimestepSeries()
        original()
    
    def when_timestep_starts(self, original, ):
        self.timeline[self.episode.timestep.index] = self.timestep
        original()
    
    def when_timestep_ends(self, original, ):
        original()

class LoggerEnhancement(Enhancement):
    """
        needs:
            EpisodeEnhancement
        creates:
            self.per_episode
            self.per_episode.average
            self.per_episode.average.reward
            self.reaction_frequency    (if self.reactions)
    """
    
    def when_mission_starts(self, original):
        # 
        # reaction_frequency
        # 
        if hasattr(self, "reactions") and type(self.reactions) != type(None):
            self.reaction_frequency = LazyDict({ each:0 for each in self.reactions })
        
        # 
        # per_episode
        #
        self.all_rewards = 0
        self.per_episode = LazyDict(
            average=LazyDict(
                reward=0,
            ),
        )
        original()
        
    
    def when_timestep_starts(self, original):
        # get a reaction
        original()
        
        # 
        # update reaction_frequency
        # 
        if hasattr(self, "reaction_frequency"):
            length_before = len(tuple(self.reaction_frequency.keys()))
            self.reaction_frequency[self.timestep.reaction] += 1
            length_after = len(tuple(self.reaction_frequency.keys()))
            if length_before < length_after:
                sort_keys(self.reaction_frequency)
    
    def when_timestep_ends(self, original):
        # 
        # rewards
        # 
        self.all_rewards    += self.timestep.reward
        self.per_episode.average.reward = self.all_rewards/len(self.episodes)
        
        original()
