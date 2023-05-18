from copy import deepcopy
from random import random, sample, choices
from time import sleep
import json
import math
import threading
from multiprocessing import Manager

import __dependencies__.blissful_basics as bb                                                   
from __dependencies__.blissful_basics import to_pure, countdown, print, singleton, FS, stringify, LazyDict
from __dependencies__.elegant_events import Server
import torch                                                                    # pip install torch
import numpy                                                                    # pip install numpy
from icecream import ic                                                         # pip install icecream
from trivial_torch_tools import to_tensor                                       # pip install trivial_torch_tools
ic.configureOutput(includeContext=True)

from config import config, path_to
from envs.warthog import WarthogEnv, WaypointEntry
from generic_tools.geometry import get_distance, get_angle_from_origin, zero_to_2pi, pi_to_pi, abs_angle_difference
from generic_tools.numpy import shift_towards
from generic_tools.hill_climbing import guess_to_maximize
from generic_tools.universe.agent import Skeleton

json.fallback_table[numpy.ndarray] = lambda array: array.tolist() # make numpy arrays jsonable
mean_squared_error_core = torch.nn.MSELoss()
mean_squared_error = lambda arg1, arg2: to_pure(mean_squared_error_core(to_tensor(arg1), to_tensor(arg2)))

# 
# establish filepaths
# 
process_id = random() # needed so that two of these can run in parallel
recorder_path       = f"{config.output_folder}/recorder.yaml" 

# for testing, setup the perfect adjuster as comparison
perfect_answer = to_tensor([
    [ 1,     0,    config.simulator.velocity_offset, ],
    [ 0,     1,        config.simulator.spin_offset, ],
    [ 0,     0,                                   1, ],
]).numpy()
perfect_transform = perfect_answer[0:2,:]

# 
# models
# 
generate_next_spacial_info = WarthogEnv.sim_warthog
generate_next_observation  = WarthogEnv.generate_observation
shared_thread_data = None

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
        
        if config.action_adjuster.always_perfect:
            self._transform = perfect_transform
        
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
    Transform.inital = perfect_transform

# TODO: replace this with a normalization method
spacial_coefficients = WarthogEnv.SpacialInformation([
    100, # x
    100, # y
    1, # angle
    0.3, # velocity # arbitraryly picked value 
    0.3, # spin # arbitraryly picked value 
])
# this class does a little data organization, then sends data to the solver, and receives answers from the solver
class ActionAdjuster:
    self = None
    def __init__(self, policy, waypoints_list, recorder=None):
        ActionAdjuster.self = self
        self.original_policy = policy
        self.policy = lambda *args, **kwargs: self.original_policy(*args, **kwargs)
        self.transform = Transform()
        self.should_update = countdown(config.action_adjuster.update_frequency)
        self.recorder = recorder            
        self.timestep = 0                   
        self.buffer_for_actual_spacial_values = []
        self.buffer_for_input_data            = []
        self.incoming_records_to_log          = []
        
        if self.recorder == None:
            from __dependencies__.rigorous_recorder import RecordKeeper
            self.recorder = RecordKeeper()
        
        self.recorder.live_write_to(recorder_path, as_yaml=True)
        
        if not config.action_adjuster.use_threading:
            self.solver = ActionAdjusterSolver(ActionAdjuster.self.policy, waypoints_list)
        else:
            self.processor_thread = threading.Thread(
                target=ActionAdjusterSolver.solver_loop,
                args=(shared_thread_data, waypoints_list)
            )
            self.processor_thread.start()
    
    def add_data(self, observation, additional_info):
        with shared_thread_data.lock:
            shared_thread_data["timestep"] = self.timestep
            sleep(0.5)
        
        self.timestep += 1
        self.recorder.add(timestep=self.timestep)
        
        self.buffer_for_input_data.append(dict(
            historic_transform=deepcopy(self.transform),
            observation=observation,
            additional_info=[
                additional_info["spacial_info_with_noise"],
                additional_info["current_waypoint_index"],
                additional_info["horizon"],
                additional_info["action_duration"],
            ],
        ))
        self.buffer_for_actual_spacial_values.append(additional_info["spacial_info_with_noise"])
        if self.should_update():
            self.send_data_to_solver()
            # run solver each update step
            if not config.action_adjuster.use_threading:
                if self.solver.receive_data_from_main_thread():
                    self.solver.fit_points()
                    self.solver.send_data_to_main_thread()
            self.receive_output_from_solver()
    
    def send_data_to_solver(self):
        # append the new data
        with shared_thread_data.lock:
            shared_thread_data["buffer_for_actual_spacial_values"] = shared_thread_data.get("buffer_for_actual_spacial_values", []) + self.buffer_for_actual_spacial_values
            shared_thread_data["buffer_for_input_data"]            = shared_thread_data.get("buffer_for_input_data",            []) + self.buffer_for_input_data
        self.buffer_for_actual_spacial_values.clear()
        self.buffer_for_input_data.clear()
        
    def receive_output_from_solver(self):
        with shared_thread_data.lock:
            self.incoming_records_to_log += shared_thread_data.get("records_to_log", [])
            shared_thread_data["records_to_log"] = []
            self.transform = Transform.from_json(
                json.loads(shared_thread_data.get("transform_json", json.dumps(self.transform)))
            )
    
    def write_pending_records(self):
        for each in self.incoming_records_to_log:
            self.recorder.commit(additional_info=each)
        self.incoming_records_to_log.clear()
        
    
class ActionAdjusterSolver:
    @staticmethod
    def solver_loop(shared_thread_data, waypoints_list):
        action_adjuster_processor = None
        while threading.main_thread().is_alive():
            # 
            # init
            # 
            if not action_adjuster_processor:
                if ActionAdjuster.self:
                    action_adjuster_processor = ActionAdjusterSolver(ActionAdjuster.self.policy, waypoints_list)
                    continue
                sleep(1)
                continue
            
            # 
            # actual main loop
            # 
            if not action_adjuster_processor.receive_data_from_main_thread(): continue
            action_adjuster_processor.fit_points()
            action_adjuster_processor.send_data_to_main_thread()

    def __init__(self, policy, waypoints_list):
        self.original_policy = policy
        self.waypoints_list = waypoints_list
        self.policy = lambda *args, **kwargs: self.original_policy(*args, **kwargs)
        self.actual_spacial_values = []
        self.input_data            = []
        self.transform             = Transform()
        self.canidate_transform    = Transform()
        self.stdev = 0.01
        self.selected_solutions = set([ self.transform ])
        self.local_buffer_for_records = []  # only the processor thread should use this attribute
        self.timestep_of_shared_info = None # only the processor thread should use this attribute
        
    def receive_data_from_main_thread(self):
        existing_data_count = len(self.input_data)
        with shared_thread_data.lock:
            self.timestep_of_shared_info = shared_thread_data.get("timestep", 0)
            self.input_data            += shared_thread_data.get("buffer_for_input_data",            [])
            self.actual_spacial_values += shared_thread_data.get("buffer_for_actual_spacial_values", [])
            shared_thread_data["buffer_for_input_data"] = []
            shared_thread_data["buffer_for_actual_spacial_values"] = []
        
        if any(each.x == each.y == 0 for each in self.actual_spacial_values[1:]):
            import code; code.interact(local={**globals(),**locals()})
        
        # if no new data
        if existing_data_count == len(self.input_data): 
            sleep(1)
            return False
        
        assert len(self.actual_spacial_values) == len(self.input_data)
        # 
        # cap the history size
        # 
        if config.action_adjuster.max_history_size < math.inf:
            if len(self.actual_spacial_values) > config.action_adjuster.max_history_size:
                self.actual_spacial_values = self.actual_spacial_values[-config.action_adjuster.max_history_size:]
                self.input_data            = self.input_data[-config.action_adjuster.max_history_size:]
        
        return True
        
    def send_data_to_main_thread(self):
        with shared_thread_data.lock:
            shared_thread_data["records_to_log"] = shared_thread_data.get("records_to_log", []) + self.local_buffer_for_records
            shared_thread_data["transform_json"] = json.dumps(self.transform)
        self.local_buffer_for_records.clear()
        
    def fit_points(self):
        # no data
        if len(self.input_data) == 0:
            return
        
        # skip the first X entries, because there is no predicted value for the first entry (predictions need source data)
        real_spacial_values = self.actual_spacial_values[config.action_adjuster.future_projection_length:]
        def objective_function(numpy_array):
            transform = Transform(numpy_array)
            spacial_projections = []
            predicted_spacial_values = []
            for each_input_data, real_spacial_value in zip(self.input_data, real_spacial_values):
                spacial_expectation, observation_expectation = self.project(
                    transform=transform,
                    real_transformation=False,
                    policy=self.policy,
                    _actual=real_spacial_value,
                    **each_input_data,
                )
                spacial_projections.append(spacial_expectation)
                predicted_spacial_values.append(spacial_expectation[-1]) 
            
            exponent = 2 if config.curve_fitting_loss == 'mean_squared_error' else 1
            # loss function
            losses = [0,0,0,0,0] # x, y, angle, velocity, spin
            # Note: len(predicted_spacial_values) should == len(real_spacial_values) + future_projection_length
            # however, it will be automatically truncated because of the zip behavior
            with print.indent:
                for each_actual, each_predicted in zip(real_spacial_values, predicted_spacial_values):
                    x1, y1, angle1, velocity1, spin1 = each_actual
                    x2, y2, angle2, velocity2, spin2 = each_predicted
                    iteration_total = 0
                    losses[0] += spacial_coefficients.x        * (abs((x1        - x2       ))          )**exponent   
                    losses[1] += spacial_coefficients.y        * (abs((y1        - y2       ))          )**exponent   
                    losses[2] += spacial_coefficients.angle    * ((abs_angle_difference(angle1, angle2)))**exponent # angle is different cause it wraps (0 == 2π)
                    losses[3] += spacial_coefficients.velocity * (abs((velocity1 - velocity2))          )**exponent   
                    losses[4] += spacial_coefficients.spin     * (abs((spin1     - spin2    ))          )**exponent   
            
                print(f'''losses = {losses}''')
            # print(f'''    {-loss}: {transform}''')
            return -sum(losses)
        
        # 
        # overfitting protection (validate the canidate)
        # 
        if not config.action_adjuster.disabled and not config.action_adjuster.always_perfect:
            solutions = list(self.selected_solutions) + [ self.canidate_transform ]
            perfect_score = objective_function(perfect_transform)
            print("")
            print(f'''perfect objective value: {perfect_score}''')
            scores = tuple(
                objective_function(each_transform.as_numpy)
                    for each_transform in solutions
            )
            distances_from_perfect = tuple(
                mean_squared_error(each_transform.as_numpy, perfect_transform)
                    for each_transform in solutions
            )
            print("evaluating transforms:")
            for each_transform, each_score, each_distance in zip(solutions, scores, distances_from_perfect):
                print(f'''    objective value: {each_score:.3f}: distance from perfect:{each_distance:.4f}: {each_transform}''')
            
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
        
        # 
        # record data
        # 
        if True:
            score_before = objective_function(self.transform.as_numpy)
            timestep = self.timestep_of_shared_info
            # kill this process once the limit is reaced
            if timestep > config.simulator.max_episode_steps:
                exit()
            
            self.local_buffer_for_records.append(dict(
                timestep=shared_thread_data["timestep"], # the timestep the computation was finished
                timestep_started=timestep, # the timestep of the data that was used for the calculation
                line_fit_score=score_before,
                is_active_transform=True,
            ))
        
        # 
        # generate new canidate
        # 
        if not config.action_adjuster.disabled and not config.action_adjuster.always_perfect:
            # find next best
            best_new_transform = Transform(
                guess_to_maximize(
                    objective_function,
                    initial_guess=self.transform.as_numpy,
                    stdev=self.stdev,
                    max_iterations=config.action_adjuster.solver_max_iterations,
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
            self.local_buffer_for_records.append(dict(canidate_transform=self.canidate_transform))
            
            score_after = objective_function(self.canidate_transform.as_numpy)
            print(f'''new canidate transform = {self.canidate_transform}''')
            print(f'''score_before   = {score_before}''')
            print(f'''canidate score = {score_after}''')
            # if no improvement at all, then shrink the stdev
            if score_after < score_before:
                self.stdev = self.stdev/10
    
    # returns twos list, one of projected spacial_info's one of projected observations
    def project(self, policy, observation, additional_info, transform=None, real_transformation=True, historic_transform=None, _actual=None):
        current_spacial_info, current_waypoint_index, horizon, action_duration = additional_info
        remaining_waypoints = self.waypoints_list[current_waypoint_index:]
        observation_expectation = []
        spacial_expectation = []
        with print.indent:
            if _actual:
                print(f'''_actual = {_actual}''')
            for each in range(config.action_adjuster.future_projection_length):
                print(f"projection_index:{each}")
                with print.indent:
                    action = policy(observation)
                    
                    # undo the effects of the at-the-time transformation
                    if historic_transform:
                        action = historic_transform.modify_action(
                            action=action,
                            reverse_transformation=real_transformation, # fight-against the new transformation
                        )
                    
                    print(f'''action before = {action}''')
                    velocity_action, spin_action = transform.modify_action(
                        action=action,
                        reverse_transformation=(not real_transformation)
                    )
                    print(f'''action after = {[velocity_action, spin_action]}''')
                    
                    next_spacial_info = generate_next_spacial_info(
                        old_spacial_info=current_spacial_info,
                        velocity_action=velocity_action,
                        spin_action=spin_action,
                        action_duration=action_duration,
                        debug=True,
                    )
                    next_spacial_info_before = generate_next_spacial_info(
                        old_spacial_info=current_spacial_info,
                        velocity_action=action[0],
                        spin_action=action[1],
                        action_duration=action_duration,
                        debug=True,
                    )
                    print(f'''next_spacial_info_before = {next_spacial_info_before}''')
                    print(f'''next_spacial_info = {next_spacial_info}''')
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
    
    def __init__(self, observation_space, reaction_space, policy, recorder, waypoints_list, **config):
        self.observation_space  = observation_space
        self.reaction_space     = reaction_space
        self.accumulated_reward = 0
        self.policy   = policy
        self.recorder = recorder
        self.action_adjuster = ActionAdjuster(policy=policy, recorder=recorder, waypoints_list=waypoints_list)

    def when_mission_starts(self):
        pass
    def when_episode_starts(self):
        pass
    def when_timestep_starts(self):
        """
        read: self.observation
        write: self.reaction = something
        """
        action = self.policy(self.timestep.observation)
        if config.action_adjuster.use_transform:
            action = self.action_adjuster.transform.modify_action(
                action
            )
        self.timestep.reaction = action
    def when_timestep_ends(self):
        """
        read: self.timestep.reward
        """
        self.accumulated_reward += self.timestep.reward
        self.recorder.add(timestep=self.timestep.index)
        self.recorder.add(accumulated_reward=self.accumulated_reward)
        self.recorder.add(reward=self.timestep.reward)
        self.recorder.commit()
        self.action_adjuster.write_pending_records()
        self.recorder.add(timestep=self.timestep.index) # encase there's another commit during the same timestep
        self.action_adjuster.add_data(self.timestep.observation, self.timestep.hidden_info)
    def when_episode_ends(self):
        pass
    def when_mission_ends(self):
        pass


@bb.run_in_main
def _():
    global shared_thread_data
    if config.action_adjuster.use_threading:
        shared_thread_data = Manager().dict()
        shared_thread_data.lock = threading.Lock()
    else:
        shared_thread_data = LazyDict()
        shared_thread_data.lock = threading.Lock()

bb.run_main_hooks_if_needed(__name__)