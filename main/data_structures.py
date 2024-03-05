import math
import time
import csv
from copy import deepcopy
from collections import namedtuple
import json

from gym import spaces
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import gym
import matplotlib as mpl
import numpy as np
import numpy

from __dependencies__ import blissful_basics as bb
from __dependencies__.blissful_basics import Csv, create_named_list_class, FS, print, stringify, clip, countdown, LazyDict
from __dependencies__.grug_test import yaml, ez_yaml, register_named_tuple
from super_hash import super_hash

class Unknown:
    pass

Action = namedtuple('Action', ['velocity', 'spin'])
StepOutput = namedtuple('StepOutput', ['observation', 'reward', 'done', 'debug'])
StepSideEffects = namedtuple('StepSideEffects', [
    'action',
    'crosstrack_error',
    'episode_steps',
    'omega_reward',
    'phi_error',
    'prev_absolute_action',
    'prev_next_waypoint_index_',
    'reward',
    'total_ep_reward',
    'vel_error',
    'vel_reward',
    'twist',
    'prev_angle',
    'pose',
    'closest_distance',
    'next_waypoint_index_',
    'ep_poses',
])
GetObservationOutput = namedtuple('GetObservationOutput', [
    'obs',
    'closest_distance',
    'next_waypoint_index_'
])
RewardOutput = namedtuple('RewardOutput', [
    "running_reward",
    "velocity_error",
    "crosstrack_error",
    "phi_error"
])
SimWarthogOutput = namedtuple('SimWarthogOutput', [
    "twist",
    "prev_angle",
    "pose",
    "ep_poses",
    "absolute_action",
])
PoseEntry = namedtuple('PoseEntry', [
    "x",
    "y",
    "angle",
])
TwistEntry = namedtuple('TwistEntry', [
    "velocity",
    "spin",
    "unknown",
])
SpacialHistory = namedtuple('SpacialHistory', [
    "x",
    "y",
    "angle",
    "velocity",
    "spin",
    "new_velocity",
    "new_spin",
])


# 
# support classes (mostly wrappers around lists to make debugging easier)
# 
SpacialInformation = namedtuple(
    "SpacialInformation",
    [ "x", "y", "angle", "velocity", "spin", "timestep" ]
)
SpacialInformation.with_timestep = lambda self, timestep: SpacialInformation(x=self.x,y=self.y,angle=self.angle,spin=self.spin,velocity=self.velocity, timestep=timestep)
ReactionClass = namedtuple(
    "ReactionClass",
    [ "relative_velocity", "relative_spin", ]
)
WaypointGap = create_named_list_class([ f"distance", f"angle_directly_towards_next", f"desired_angle_at_next", f"velocity" ])

@yaml.register_class
class Waypoint(numpy.ndarray):
    yaml_tag = "!python/warthog/Waypoint"
    keys = [ "x", "y", "angle", "velocity" ]
    
    def __new__(cls, data):
        # note: https://stackoverflow.com/questions/7342637/how-to-subclass-a-subclass-of-numpy-ndarray?rq=4
        return numpy.asarray(data).view(cls)
        
    @property
    def x(self): return self[0]
    @x.setter
    def x(self, value): self[0] = value
    
    @property
    def y(self): return self[1]
    @y.setter
    def y(self, value): self[1] = value
    
    @property
    def angle(self): return self[2]
    @angle.setter
    def angle(self, value): self[2] = value
    
    @property
    def velocity(self): return self[3]
    @velocity.setter
    def velocity(self, value): self[3] = value
    
    def __repr__(self):
        return f'''Waypoint(x:{f"{self.x:.5f}".rjust(9)}, y:{f"{self.y:.5f}".rjust(9)}, angle:{f"{self.angle:.5f}".rjust(9)}, velocity:{f"{self.velocity:.5f}".rjust(9)})'''
    
    @classmethod
    def from_yaml(cls, constructor, node):
        data = json.loads(node.value)
        return cls([data["x"],data["y"],data["angle"],data["velocity"]])
    
    @classmethod
    def to_yaml(cls, representer, object_of_this_class):
        representation = json.dumps(dict(
            x=object_of_this_class.x,
            y=object_of_this_class.y,
            angle=object_of_this_class.angle,
            velocity=object_of_this_class.velocity,
        ))
        # ^ needs to be a string (or some other yaml-primitive)
        return representer.represent_scalar(
            tag=cls.yaml_tag,
            value=representation,
            style=None,
            anchor=None
        )
    
    def __hash__(self):
        return hash(
            json.dumps(dict(
                x=self.x,
                y=self.y,
                angle=self.angle,
                velocity=self.velocity,
            ))
        )

@yaml.register_class
class Observation:
    yaml_tag = "!python/warthog/Observation"
    
    def __init__(self, values=None):
        self.timestep = None
        self.absolute_velocity = None
        self.absolute_spin     = None
        self.waypoint_gaps = []
        if bb.is_iterable(values):
            values = list(values)
            self.timestep = values.pop(-1)
            self.absolute_velocity = values.pop(-1)
            self.absolute_spin     = values.pop(-1)
            while len(values) > 0:
                waypoint_gap = []
                for index in range(len(WaypointGap.names_to_index)):
                    waypoint_gap.append(values.pop(-1))
                self.waypoint_gaps.append(
                    WaypointGap(
                        reversed(waypoint_gap)
                    )
                )
        else:
            raise Exception(f'''Observation() got non-iterable argument''')
    
    def __len__(self):
        return len(self.__json__())
    
    def __iter__(self):
        return iter(self.to_numpy())
    
    def __json__(self):
        output = []
        for each_waypoint_gap in reversed(self.waypoint_gaps):
            for each_value in each_waypoint_gap:
                output.append(each_value)
        output.append(self.absolute_spin)
        output.append(self.absolute_velocity)
        output.append(self.timestep)
        return output
    
    def to_numpy(self):
        as_list = self.__json__()
        as_list.pop(-1)
        return numpy.array(as_list)
    
    def __hash__(self):
        return hash(super_hash(self.__repr__()))
    
    def __repr__(self):
        """
            Note:
                this function is used in the hash method, so the number of decimals printed does matter significantly for determining equality
        """
        return f"""Observation(timestep={self.timestep}, absolute_velocity={f"{self.absolute_velocity:0.7f}".ljust(9,"0")}, absolute_spin={f"{self.absolute_spin:0.7f}".ljust(9,"0")}, waypoint_gaps={self.waypoint_gaps})"""
    
    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(json.loads(node.value))
    
    @classmethod
    def to_yaml(cls, representer, object_of_this_class):
        representation = json.dumps(object_of_this_class)
        # ^ needs to be a string (or some other yaml-primitive)
        return representer.represent_scalar(
            tag=cls.yaml_tag,
            value=representation,
            style=None,
            anchor=None
        )

AdditionalInfo = namedtuple(
    "AdditionalInfo",
    [
        # chronologically
        "timestep_index",
        "action_duration",
        "spacial_info",
        "spacial_info_with_noise",
        "observation_from_spacial_info_with_noise",
        "historic_transform",
        "original_reaction",
        "mutated_relative_reaction",
        "next_spacial_info",
        "next_spacial_info_spacial_info_with_noise",
        "next_observation_from_spacial_info_with_noise",
        "next_closest_index",
        "reward",
    ]
)

for each in [Action, StepOutput, StepSideEffects, GetObservationOutput, RewardOutput, SimWarthogOutput, PoseEntry, TwistEntry, SpacialHistory, SpacialInformation, ReactionClass, AdditionalInfo ]:
    register_named_tuple(each, f"{each.__name__}")
