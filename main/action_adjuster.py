import json
from copy import deepcopy
from time import sleep
import math
import threading

import torch                                                                    # pip install torch
import numpy                                                                    # pip install numpy
import blissful_basics as bb                                                    # pip install blissful_basics
from blissful_basics import to_pure, countdown, print, singleton, FS, stringify # pip install blissful_basics
from icecream import ic                                                         # pip install icecream
from trivial_torch_tools import to_tensor                                       # pip install trivial_torch_tools
import json_fix
ic.configureOutput(includeContext=True)

from config import config, path_to
from envs.warthog import WarthogEnv, WaypointEntry
from generic_tools.geometry import get_distance, get_angle_from_origin, zero_to_2pi, pi_to_pi, abs_angle_difference
from generic_tools.numpy import shift_towards
from generic_tools.hill_climbing import guess_to_maximize
from generic_tools.universe.agent import Skeleton

import numpy
json.fallback_table[numpy.ndarray] = lambda array: array.tolist() # make numpy arrays jsonable

perfect_answer = to_tensor([
    [ 1,     0,    config.simulator.velocity_offset, ],
    [ 0,     1,        config.simulator.spin_offset, ],
    [ 0,     0,                                   1, ],
]).numpy()

# init/reset the files
FS.write(path=path_to.action_adjuster_data_buffer, data='[]')
FS.write(path=path_to.action_adjuster_transform_file, data='')
with open(path_to.action_adjuster_transform_file, 'w') as outfile:
    json.dump(perfect_answer, outfile)

# 
# models
# 
generate_next_spacial_info = WarthogEnv.sim_warthog
generate_next_observation  = WarthogEnv.generate_observation

recorder_path = f"{path_to.default_output_folder}/recorder.yaml"

class Transform:
    inital = numpy.eye(config.simulator.action_length+1)[0:2,:]
    
    # think of this as a "from numpy array" method
    def __init__(self, value=None):
        self.inital_arg = value
        if isinstance(value, type(None)):
            self._transform = to_tensor(self.inital).numpy()
        elif isinstance(value, numpy.ndarray):
            self._transform = value
        else:
            self._transform = to_tensor(value).numpy()
        
    def __deepcopy__(self, arg1):
        return Transform(to_tensor(self._transform).numpy())
    
    def __repr__(self):
        rows = [
            "[ " + ", ".join([ f"{each_cell:.3f}" for each_cell in each_row ]) + " ]"
                for each_row in to_pure(self._transform)
        ]
        return "[ " + ", ".join([ each_row for each_row in rows]) + " ]"
    
    def __json__(self):
        return self._transform.tolist()
    
    @staticmethod
    def from_json(json_parsed):
        return Transform(json_parsed)
    
    @property
    def as_numpy(self):
        return self._transform
    
    def modify_action(self, action, reverse_transformation=False):
        # the full transform needs to be 3x3 with a constant bottom-row of 0,0,1
        transform = numpy.vstack([
            self._transform[0],
            self._transform[1],
            to_tensor([0,0,1]).numpy()
        ])
        
        # the curve-fitter is finding the transform that would make the observed-trajectory match the model-predicted trajectory
        # (e.g. what transform is the world doing to our actions; once we know that we can compensate with and equal-and-opposite transformation)
        if reverse_transformation:
            *result, constant = numpy.inner(
                to_tensor([*action, 1]).numpy(),
                transform,
            )
            return to_tensor(result).numpy()
        # the real transformation needs to compensate (inverse of curve-fitter transform)
        else:
            inverse_transform = numpy.linalg.inv( to_tensor(transform).numpy() )
            *result, constant = numpy.inner(
                to_tensor([*action, 1]).numpy(),
                inverse_transform,
            )
            return to_tensor(result).numpy()
        
    
    def __hash__(self):
        return hash((id(numpy.ndarray), self._transform.shape, tuple(each for each in self._transform.flat)))

if config.action_adjuster.default_to_perfect:
    Transform.inital = perfect_answer[0:2,:]

# TODO: replace this with a normalization method
spacial_coefficients = WarthogEnv.SpacialInformation([
    1, # x
    1, # y
    1, # angle
    0.3, # velocity # arbitraryly picked value 
    0.3, # spin # arbitraryly picked value 
])
# TODO: combine this class with the ActionAdjustedAgent class
class ActionAdjuster:
    self = None
    waypoints_list = []
    def __init__(self, policy, initial_transform=None, recorder=None):
        ActionAdjuster.self = self
        self.original_policy = policy
        self.policy = lambda *args, **kwargs: self.original_policy(*args, **kwargs)
        self.actual_spatial_values = []
        self.input_data            = []
        self.transform             = Transform()
        self.canidate_transform    = Transform()
        self.should_update = countdown(config.action_adjuster.update_frequency)
        self.stdev = 0.01
        self.selected_solutions = set([ self.transform ])
        self.recorder = recorder
        self.timestep = 0
        self.buffer_for_write = []
        
        if self.recorder == None:
            from rigorous_recorder import RecordKeeper
            self.recorder = RecordKeeper()
        
        self.recorder.live_write_to(recorder_path, as_yaml=True)
        
        self.processor_thread = threading.Thread(target=processor, args=())
        self.processor_thread.start()
    
    def write_to_data_buffer(self):
        buffer = []
        try:
            with open(path_to.action_adjuster_data_buffer, 'r') as in_file:
                buffer = json.load(in_file)
        except Exception as error:
            pass
        
        for observation, additional_info in self.buffer_for_write:
            buffer.append((observation, dict(
                spacial_info_with_noise=additional_info["spacial_info_with_noise"],
                current_waypoint_index=additional_info["current_waypoint_index"],
                horizon=additional_info["horizon"],
                action_duration=additional_info["action_duration"],
            )))
        self.buffer_for_write.clear()
        
        FS.write(data=json.dumps(buffer), path=path_to.action_adjuster_data_buffer)
    
    def read_from_buffer(self):
        buffer = []
        try:
            with open(path_to.action_adjuster_data_buffer, 'r') as in_file:
                buffer = json.load(in_file)
        except Exception as error:
            pass
        FS.write(data='[]', path=path_to.action_adjuster_data_buffer)
        
        # add back class data
        for observation, additional_info in buffer:
            additional_info["spacial_info_with_noise"] = WarthogEnv.SpacialInformation(additional_info["spacial_info_with_noise"])
            
        return buffer
    
    def write_transform(self):
        print("writing transform")
        try:
            with open(path_to.action_adjuster_transform_file, 'w') as out_file:
                json.dump(self.transform, out_file)
        except Exception as error:
            print(f'''error = {error}''')
    
    def read_transform(self):
        try:
            with open(path_to.action_adjuster_transform_file, 'r') as in_file:
                self.transform = Transform.from_json(
                    json.load(in_file)
                )
        except Exception as error:
            print(f"note, unable to read transform data, {error}")
    
    def add_data(self, observation, additional_info, is_processor_thread=False):
        self.timestep += 1
        self.recorder.add(timestep=self.timestep)
        
        if not is_processor_thread:
            self.buffer_for_write.append((observation, additional_info))
            if self.should_update():
                self.write_to_data_buffer()
                self.read_transform()
        else:
            self.input_data.append(dict(
                policy=self.policy,
                historic_transform=deepcopy(self.transform),
                observation=observation,
                additional_info=[
                    additional_info["spacial_info_with_noise"],
                    additional_info["current_waypoint_index"],
                    additional_info["horizon"],
                    additional_info["action_duration"],
                ],
            ))
            self.actual_spatial_values.append(additional_info["spacial_info_with_noise"])
            # NOTE: this^ is only grabbing the last, it would be valid to grab all of them
            # future work here might try grabbing all of them, or reducing the length, especially if the model/policy is thought to be bad/noisy
            assert len(self.actual_spatial_values) == len(self.input_data)
            # cap the history size
            if config.action_adjuster.max_history_size < math.inf:
                if len(self.actual_spatial_values) > config.action_adjuster.max_history_size:
                    self.actual_spatial_values = self.actual_spatial_values[-config.action_adjuster.max_history_size:]
                    self.input_data            = self.input_data[-config.action_adjuster.max_history_size:]
            
            action = self.policy(observation)
            
            if self.should_update():
                self.fit_points()
    
    def fit_points(self):
        # no data
        if len(self.input_data) == 0:
            return
        
        # skip the first X entries, because there is no predicted value for the first entry (predictions need source data)
        real_spatial_values = self.actual_spatial_values[config.action_adjuster.future_projection_length:]
        def objective_function(numpy_array):
            transform = Transform(numpy_array)
            predicted_spatial_values = []
            for each_input_data in self.input_data:
                spacial_expectation, observation_expectation = self.project(
                    transform=transform,
                    real_transformation=False,
                    **each_input_data,
                )
                predicted_spatial_values.append(spacial_expectation[-1]) 
            
            # loss function
            loss = 0
            # Note: len(predicted_spatial_values) should == len(real_spatial_values) + future_projection_length
            # however, it will be automatically truncated because of the zip behavior
            for each_actual, each_predicted in zip(real_spatial_values, predicted_spatial_values):
                x1, y1, angle1, velocity1, spin1 = each_actual
                x2, y2, angle2, velocity2, spin2 = each_predicted
                iteration_total = 0
                iteration_total += spacial_coefficients.x        * abs((x1        - x2       ))
                iteration_total += spacial_coefficients.y        * abs((y1        - y2       ))
                iteration_total += spacial_coefficients.angle    * (abs_angle_difference(angle1, angle2)) # angle is different cause it wraps (0 == 2π)
                iteration_total += spacial_coefficients.velocity * abs((velocity1 - velocity2))
                iteration_total += spacial_coefficients.spin     * abs((spin1     - spin2    ))
                
                loss += iteration_total
            # print(f'''    {-loss}: {transform}''')
            return -loss
        
        # 
        # overfitting protection (validate the canidate)
        # 
        if not config.action_adjuster.disabled:
            solutions = list(self.selected_solutions) + [ self.canidate_transform ]
            scores = tuple(
                objective_function(each_transform.as_numpy)
                    for each_transform in solutions
            )
            for each_transform, each_score in zip(solutions, scores):
                print(f'''    {each_score}: {each_transform}''')
            
            best_with_new_data = bb.arg_maxs(
                args=solutions,
                values=scores,
            )
            new_data_invalidated_recent_best = self.canidate_transform not in best_with_new_data
            # basically overfitting detection
            # reduce stdev, and don't use the canidate
            if new_data_invalidated_recent_best:
                self.stdev = self.stdev/2
                # choose the long-term best as the starting point
                self.transform = best_with_new_data[0]
            else:
                print(f'''canidate passed inspection: {self.canidate_transform}''')
                print(f'''prev transform            : {self.transform}''')
                print(f'''canidate score: {objective_function(self.canidate_transform.as_numpy)}''')
                print(f'''prev score    : {objective_function(self.transform.as_numpy)}''')
                self.selected_solutions.add(self.canidate_transform)
                # use the canidate transform as the base for finding new answers
                self.transform = self.canidate_transform
        
        if True:
            score_before     = objective_function(self.transform.as_numpy)
            self.recorder.add(line_fit_score=score_before)
            self.recorder.commit()
        
        # 
        # generate new canidate
        # 
        if not config.action_adjuster.disabled:
            # find next best
            best_new_transform = Transform(
                guess_to_maximize(
                    objective_function,
                    initial_guess=self.transform.as_numpy,
                    stdev=self.stdev
                )
            )
            
            # canidate is the incremental shift towards next_best
            self.canidate_transform = Transform(
                shift_towards(
                    new_value=best_new_transform.as_numpy,
                    old_value=self.transform.as_numpy,
                    proportion=config.action_adjuster.update_rate,
                )
            )
            
            score_after = objective_function(self.canidate_transform.as_numpy)
            print(f'''new canidate transform = {self.canidate_transform}''')
            print(f'''score_before   = {score_before}''')
            print(f'''canidate score = {score_after}''')
            # if no improvement at all, then shrink the stdev
            if score_after < score_before:
                self.stdev = self.stdev/10
    
    # returns twos list, one of projected spacial_info's one of projected observations
    def project(self, policy, observation, additional_info, transform=None, real_transformation=True, historic_transform=None):
        current_spatial_info, current_waypoint_index, horizon, action_duration = additional_info
        remaining_waypoints = ActionAdjuster.waypoints_list[current_waypoint_index:]
        observation_expectation = []
        spacial_expectation = []
        for each in range(config.action_adjuster.future_projection_length):
            action = policy(observation)
            
            # undo the effects of the at-the-time transformation
            if historic_transform:
                action = historic_transform.modify_action(
                    action=action,
                    reverse_transformation=real_transformation, # fight-against the new transformation
                )
            
            velocity_action, spin_action = transform.modify_action(
                action=action,
                reverse_transformation=(not real_transformation)
            )
            
            next_spacial_info = generate_next_spacial_info(
                old_spatial_info=current_spatial_info,
                velocity_action=velocity_action,
                spin_action=spin_action,
                action_duration=action_duration,
            )
            next_observation = generate_next_observation(
                remaining_waypoints=remaining_waypoints,
                horizon=horizon,
                current_spacial_info=next_spacial_info,
            )
            spacial_expectation.append(next_spacial_info)
            observation_expectation.append(next_observation)
            
            observation          = next_observation
            current_spacial_info = next_spacial_info
        
        return spacial_expectation, observation_expectation

class ActionAdjustedAgent(Skeleton):
    previous_timestep = None
    timestep          = None
    next_timestep     = None
    # timestep attributes:
    # self.timestep.index
    # self.timestep.observation
    # self.timestep.is_last_step
    # self.timestep.reward
    # self.timestep.hidden_info
    # self.timestep.reaction
    
    def __init__(self, observation_space, reaction_space, policy, recorder, **config):
        self.observation_space  = observation_space
        self.reaction_space     = reaction_space
        self.accumulated_reward = 0
        self.policy   = policy
        self.recorder = recorder
        self.action_adjuster = ActionAdjuster(policy=policy, recorder=recorder)

    def when_mission_starts(self):
        pass
    def when_episode_starts(self):
        pass
    def when_timestep_starts(self):
        """
        read: self.observation
        write: self.reaction = something
        """
        self.timestep.reaction = self.action_adjuster.transform.modify_action(
            self.policy(self.timestep.observation)
        )
    def when_timestep_ends(self):
        """
        read: self.timestep.reward
        """
        self.accumulated_reward += self.timestep.reward
        self.recorder.add(timestep=self.timestep.index)
        self.recorder.add(accumulated_reward=self.accumulated_reward)
        self.recorder.add(reward=self.timestep.reward)
        self.action_adjuster.add_data(self.timestep.observation, self.timestep.hidden_info)
    def when_episode_ends(self):
        pass
    def when_mission_ends(self):
        pass

# runs in parallel, reading the observation data, computing a better transform, and writing the transform to a file that the main thread reads
def processor():
    while True:
        if not ActionAdjuster.self:
            sleep(1)
            continue
        
        self = ActionAdjuster.self
    
        new_data = self.read_from_buffer()
        if len(new_data) == 0:
            sleep(1)
            continue
            
        for observation, additional_info in new_data:
            self.add_data(observation, additional_info, is_processor_thread=True)
            
        self.write_transform()