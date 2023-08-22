from copy import deepcopy
from random import random, sample, choices
from time import sleep, time
import json
import math
import threading
from multiprocess import Manager
from collections import namedtuple

import torch
import numpy

import __dependencies__.blissful_basics as bb
from __dependencies__.blissful_basics import to_pure, print, countdown, singleton, FS, stringify, LazyDict, create_named_list_class, Timer, Time
from __dependencies__.trivial_torch_tools import to_tensor
from __dependencies__.super_hash import super_hash
from __dependencies__.rigorous_recorder import RecordKeeper

from config import config, path_to, debug
from envs.warthog import WarthogEnv
from generic_tools.geometry import get_distance, get_angle_from_origin, zero_to_2pi, pi_to_pi, abs_angle_difference
from generic_tools.numpy import shift_towards
from generic_tools.hill_climbing import guess_to_maximize
from generic_tools.universe.agent import Skeleton

json.fallback_table[numpy.ndarray] = lambda array: array.tolist() # make numpy arrays jsonable
mean_squared_error_core = torch.nn.MSELoss()
mean_squared_error = lambda arg1, arg2: to_pure(mean_squared_error_core(to_tensor(arg1), to_tensor(arg2)))
pprint = lambda *args, **kwargs: bb.print(*(stringify(each) for each in args), **kwargs)

# 
# simple vars
# 
if True:
    recorder_path = f"{config.output_folder}/recorder.yaml" 
    time_slowdown = config.simulator.action_duration # the bigger this is, the more iterations the solver will complete before the episode is over
                    # (solver runs as fast as possible, so slowing down the main thread makes it complete relatively more iterations)
    shared_thread_data = None
        # ^ will contain
            # "timestep": an int that is always the number of the latest timestep
            # "transform_json"
            # "timestep_data"
            # "records_to_log"

    # TODO: replace this with a normalization method
    spacial_coefficients = WarthogEnv.SpacialInformation(
        x=100, 
        y=100, 
        angle=1, 
        velocity=0.3, 
        spin=0.3, 
        timestep=0 
    )

    # for testing, setup the perfect adjuster as comparison
    perfect_answer = to_tensor([
        [ 1,     0,    config.simulator.velocity_offset, ],
        [ 0,     1,        config.simulator.spin_offset, ],
        [ 0,     0,                                   1, ],
    ]).numpy()
    perfect_transform_input = perfect_answer[0:2,:]

    inital_transform = numpy.eye(config.simulator.action_length+1)[0:2,:] 
    if config.action_adjuster.default_to_perfect:
        inital_transform = perfect_transform_input

class Transform:
    inital = inital_transform
    
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
            self._transform = perfect_transform_input
        
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
    
    def adjust_action(self, action, mimic_adversity=False):
        # the full transform needs to be 3x3 with a constant bottom-row of 0,0,1
        transform = numpy.vstack([
            self._transform[0],
            self._transform[1],
            to_tensor([0,0,1]).numpy()
        ])
        
        # the curve-fitter is finding the transform that would make the observed-trajectory match the model-predicted trajectory
        # (e.g. what transform is the world doing to our actions; once we know that we can compensate with and equal-and-opposite transformation)
        # for an optimal transform, adjust_action(action, mimic_adversity=True) == env.step(action).mutated_action
        # if mimic_adversity:
        #     *result, constant = numpy.inner(
        #         to_tensor([*action, 1]).numpy(),
        #         transform,
        #     )
        #     return to_tensor(result).numpy()
        
        # the real transformation needs to compensate (inverse of curve-fitter transform)
        # e.g. env.step(adjust_action(action)).mutated_action == action
        if not mimic_adversity:
            transform = numpy.linalg.inv( to_tensor(transform).numpy() )
            
        *result, constant = numpy.inner(
            to_tensor([*action, 1]).numpy(),
            transform,
        )
        return to_tensor(result).numpy()
        
    
    def __hash__(self):
        return hash((id(numpy.ndarray), self._transform.shape, tuple(each for each in self._transform.flat)))

# this class does a little data organization, receives data from the agent, solves, then sends data back the agent (using shared_thread_data)
class Solver:
    self = None
    def __init__(self, policy, waypoints_list):
        Solver.self = self
        self.stdev = config.cmaes.inital_stdev # for cmaes
        self.unconfirmed_transform      = Transform()
        self.latest_confirmed_transform = Transform()
        self.selected_solutions         = set([ self.latest_confirmed_transform ])
        self.waypoints_list             = waypoints_list
        self.policy                     = policy
        self.start_timestep             = shared_thread_data.get("timestep", 0)
        self.last_canidate_was_valid    = True # attribute is for logging purposes only
        self.solve_time                 = 0
        self.objective_func_call_count  = 0
    
    # this function is only called if multithreading is enabled
    @staticmethod
    def thread_solver_loop():
        while True:
            if Solver.self:
                Solver.self.fit_points()
    
    def fit_points(self):
        timestep_data = shared_thread_data["timestep_data"][-config.action_adjuster.max_history_size:]
        sample_size = len(timestep_data)
        # not enough data
        if sample_size < (config.action_adjuster.future_projection_length + config.action_adjuster.update_frequency):
            print("not enough data for fit_points() just yet")
            sleep(1)
            return
        
        # row1 = x, row2 = y, row3 = angle, row4 = velocity, row5 = spin, row6 = time
        correct_next_spacial_predictions = to_tensor(each.next_spacial_info for each in timestep_data).transpose(0,1)
        x,y,angle,velocity,spin,*_ = tuple(range(len(correct_next_spacial_predictions[0])))
        indicies = [x,y,angle,velocity,spin]
        losses = to_tensor([0]*len(indicies))
        
        call_count = 0
        duration = 0
        # create inputs and predictions
        def objective_function(numpy_array):
            hypothetical_transform = Transform(numpy_array)
            predicted_next_spacial_values    = tuple(
                self.project(
                    spacial_info_before_action=each.spacial_info_with_noise,
                    most_recent_action=each.original_reaction,
                    historic_transform=each.historic_transform,
                    transform=hypothetical_transform,
                    action_duration=each.action_duration,
                )
                    for each in timestep_data
            )
            # available data:
            #     timestep_data[0].timestep_index
            #     timestep_data[0].spacial_info
            #     timestep_data[0].spacial_info_with_noise
            #     timestep_data[0].observation_from_spacial_info_with_noise
            #     timestep_data[0].original_reaction
            #     timestep_data[0].historic_transform
            #     timestep_data[0].mutated_reaction
            #     timestep_data[0].next_spacial_info
            #     timestep_data[0].next_spacial_info_spacial_info_with_noise
            #     timestep_data[0].next_observation_from_spacial_info_with_noise
            #     timestep_data[0].reward
            
            exponent = 2 if config.curve_fitting_loss == 'mean_squared_error' else 1
            # loss function
            losses = [0,0,0,0,0] # x, y, angle, velocity, spin
            # Note: len(predicted_spacial_values) should == len(real_spacial_values) + future_projection_length
            # however, it will be automatically truncated because of the zip behavior
            with print.indent:
                for each_actual, each_predicted in zip(correct_next_spacial_predictions, predicted_next_spacial_values):
                    x1, y1, angle1, velocity1, spin1, *_ = each_actual
                    x2, y2, angle2, velocity2, spin2, *_ = each_predicted
                    losses[0] += spacial_coefficients.x        * (abs((x1        - x2       ))          )**exponent   
                    losses[1] += spacial_coefficients.y        * (abs((y1        - y2       ))          )**exponent   
                    losses[2] += spacial_coefficients.angle    * ((abs_angle_difference(angle1, angle2)))**exponent # angle is different cause it wraps (0 == 2π)
                    losses[3] += spacial_coefficients.velocity * (abs((velocity1 - velocity2))          )**exponent   
                    losses[4] += spacial_coefficients.spin     * (abs((spin1     - spin2    ))          )**exponent   
            
            score = -sum(losses)
            # if not globals().get("skip", None):
            #     import code; code.interact(local={**globals(),**locals()})
                
            return score
        
        # 
        # overfitting protection (validate the canidate)
        # 
        if not config.action_adjuster.disabled and not config.action_adjuster.always_perfect:
            solutions = list(self.selected_solutions) + [ self.unconfirmed_transform ]
            perfect_score = objective_function(perfect_transform_input)
            print("")
            print(f'''perfect objective value: {perfect_score}''')
            scores = tuple(
                objective_function(each_transform.as_numpy)
                    for each_transform in solutions
            )
            distances_from_perfect = tuple(
                mean_squared_error(each_transform.as_numpy, perfect_transform_input)
                    for each_transform in solutions
            )
            print("evaluating transforms:")
            for each_transform, each_score, each_distance in zip(solutions, scores, distances_from_perfect):
                print(f'''    objective value: {each_score:.3f}: distance from perfect:{each_distance:.4f}: {each_transform}''')
            
            best_with_new_data = bb.arg_maxs(
                args=solutions,
                values=scores,
            )
            new_data_invalidated_recent_best = self.unconfirmed_transform not in best_with_new_data
            self.last_canidate_was_valid = not new_data_invalidated_recent_best
            # basically overfitting detection
            # reduce stdev, and don't use the canidate
            print(f'''new_data_invalidated_recent_best: {new_data_invalidated_recent_best}''')
            if new_data_invalidated_recent_best:
                self.stdev = self.stdev/config.cmaes.reduction_rate
                # choose the long-term best as the starting point
                self.latest_confirmed_transform = best_with_new_data[0]
            else:
                self.stdev += self.stdev * config.cmaes.increase_rate
                print(f'''canidate passed inspection: {self.unconfirmed_transform}''')
                print(f'''prev transform            : {self.latest_confirmed_transform}''')
                print(f'''canidate score: {objective_function(self.unconfirmed_transform.as_numpy)}''')
                print(f'''prev score    : {objective_function(self.latest_confirmed_transform.as_numpy)}''')
                self.selected_solutions.add(self.unconfirmed_transform)
                # use the canidate transform as the base for finding new answers
                self.latest_confirmed_transform = self.unconfirmed_transform
            
        # 
        # record data
        # 
        if True:
            # send to agent
            shared_thread_data["transform_json"] = json.dumps(self.latest_confirmed_transform)
            score_before = objective_function(self.latest_confirmed_transform.as_numpy)
            # kill this process once the limit is reaced
            with print.indent: 
                if shared_thread_data["timestep"] > config.simulator.max_number_of_timesteps_per_episode:
                    print("exiting process because timestep > config.simulator.max_number_of_timesteps_per_episode")
                    exit()
                
                # dont log directly from this thread, send it to the main thread so theres not a race condition on the output file
                shared_thread_data["records_to_log"] = shared_thread_data["records_to_log"] + [dict(
                    timestep=shared_thread_data["timestep"], # the timestep the computation was finished
                    timestep_started=self.start_timestep, # the active timestep when the fit_points was called
                    line_fit_score=score_before,
                    last_canidate_was_valid=self.last_canidate_was_valid,
                    sample_size=sample_size,
                    call_count=self.objective_func_call_count,
                    fit_points_time_seconds=self.solve_time,
                    distance_to_optimal=mean_squared_error(self.latest_confirmed_transform.as_numpy, perfect_transform_input),
                    latest_confirmed_transform=to_pure(self.latest_confirmed_transform.as_numpy),
                    perfect_transform_input=to_pure(perfect_transform_input),
                )]
        
        # 
        # generate new canidate
        # 
        self.start_timestep = shared_thread_data["timestep"]
        if not config.action_adjuster.disabled and not config.action_adjuster.always_perfect:
            # find next best
            best_new_transform = Transform(
                guess_to_maximize(
                    objective_function,
                    initial_guess=self.latest_confirmed_transform.as_numpy,
                    stdev=self.stdev,
                    max_iterations=config.action_adjuster.solver_max_iterations,
                )
            )
            # print(f'''call_count = {call_count}''')
            # print(f'''duration = {duration}''')
            # print(f'''duration/call = {duration/call}''')
            # canidate is the incremental shift towards next_best
            self.unconfirmed_transform = Transform(
                shift_towards(
                    new_value=best_new_transform.as_numpy,
                    old_value=self.latest_confirmed_transform.as_numpy,
                    proportion=config.action_adjuster.update_rate,
                )
            )
            with print.indent: 
                shared_thread_data["records_to_log"] = shared_thread_data["records_to_log"] + [dict(timestep=shared_thread_data["timestep"], canidate_transform=self.unconfirmed_transform,)]
            
            score_after = objective_function(self.unconfirmed_transform.as_numpy)
            print(f'''new canidate transform = {self.unconfirmed_transform}''')
            print(f'''score_before   = {score_before}''')
            print(f'''canidate score = {score_after}''')
            # if no improvement at all, then shrink the stdev
            if score_after < score_before:
                self.stdev = self.stdev/config.cmaes.reduction_rate
                print(f'''NO IMPROVEMENT: stdev is now: {self.stdev}''')
            
        
        self.objective_func_call_count = call_count
    
    @staticmethod
    def project(
        transform,
        spacial_info_before_action,
        most_recent_action,
        historic_transform,
        action_duration,
    ):
        # action + nothing            = predicted_observation -1.0
        # action + historic transform = predicted_observation -0.7 # <- this is the "what we recorded" and the "what we have to compare against"
        # action + best transform     = predicted_observation  0.3 # <- bad b/c it'll be penalized for the 0.3, (should be 0.0) but the 0.3 
        #                                                               is only there because the historic transform was doing part of the work
        # so we need to do action - historic_transform + best transform
        if historic_transform:
            policy_action = historic_transform.adjust_action(
                action=most_recent_action,
                mimic_adversity=True, # redo the historic transform in order to 
            )
        
        # keep
        # this part is trying to guess/recreate the advesarial part of the .step() function
        relative_velocity_action, relative_spin_action = transform.adjust_action(
            action=policy_action,
            mimic_adversity=True, # we want to undo the adversity when projecting into the future
        )
        # keep
        next_spacial_info = WarthogEnv.generate_next_spacial_info(
            old_spacial_info=spacial_info_before_action,
            relative_velocity=relative_velocity_action,
            relative_spin=relative_spin_action,
            action_duration=action_duration,
        )
        # keep
        return next_spacial_info

    
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
    
    def __init__(self, observation_space, reaction_space, policy, waypoints_list, recorder=None, **options):
        self.observation_space  = observation_space
        self.reaction_space     = reaction_space
        self.accumulated_reward = 0
        self.policy   = policy
        self.recorder = recorder if recorder != None else RecordKeeper()
        
        self.active_transform = Transform()
        self.should_update_solution = countdown(config.action_adjuster.update_frequency)
        self.recorder = recorder if recorder != None else RecordKeeper()
        self.recorder.live_write_to(recorder_path, as_yaml=True)
        
        shared_thread_data["timestep"] = 0
        shared_thread_data["timestep_data"] = []
        shared_thread_data["records_to_log"] = []
        self.records_from_the_solver_thread = []
        # create the solver
        self.solver = Solver(policy=policy, waypoints_list=waypoints_list)
        
        if config.action_adjuster.use_threading:
            # start the thread
            threading.Thread(target=Solver.thread_solver_loop).start()

    def when_mission_starts(self):
        pass
    
    def when_episode_starts(self):
        action1 = self.policy(self.next_timestep.observation)
        action2 = self.policy(self.next_timestep.observation)
        assert super_hash(action1) == super_hash(action2), "When using ActionAdjustedAgent, the policy needs to be deterministic, and the given policy seems to not be"
    
    def when_timestep_starts(self):
        """
        read: self.observation
        write: self.reaction = something
        """
        vanilla_action = self.policy(self.timestep.observation)
        adjusted_action = vanilla_action
        if config.action_adjuster.use_transform:
            adjusted_action = self.active_transform.adjust_action(
                adjusted_action
            )
        self.timestep.reaction = adjusted_action
    
    def when_timestep_ends(self):
        """
        read: self.timestep.reward
        """
        # 
        # logging / record-keeping
        # 
        self.accumulated_reward += self.timestep.reward
        self.recorder.add(timestep=self.timestep.index)
        self.recorder.add(accumulated_reward=self.accumulated_reward)
        self.recorder.add(reward=self.timestep.reward)
        self.recorder.commit()
        for each in self.records_from_the_solver_thread:
            self.recorder.commit(additional_info=each)
        self.records_from_the_solver_thread.clear()
        self.recorder.add(timestep=self.timestep.index) # encase there's another commit during the same timestep
        
        # 
        # give data to solver thread
        # 
        timestep = self.timestep
        with print.indent:
            shared_thread_data["timestep"] = timestep.index
            self.recorder.add(timestep=shared_thread_data["timestep"])
            additional_info = timestep.hidden_info
            shared_thread_data["timestep_data"] = shared_thread_data["timestep_data"] + [
                # fill in the timestep index and the historic_transform, but otherwise preserve data
                WarthogEnv.AdditionalInfo(
                    timestep_index=timestep.index,
                    action_duration=additional_info.action_duration,
                    spacial_info=additional_info.spacial_info,
                    spacial_info_with_noise=additional_info.spacial_info_with_noise,
                    observation_from_spacial_info_with_noise=additional_info.observation_from_spacial_info_with_noise,
                    original_reaction=additional_info.original_reaction,
                    historic_transform=deepcopy(self.active_transform),
                    mutated_reaction=additional_info.mutated_reaction,
                    next_spacial_info=additional_info.next_spacial_info,
                    next_spacial_info_spacial_info_with_noise=additional_info.next_spacial_info_spacial_info_with_noise,
                    next_observation_from_spacial_info_with_noise=additional_info.next_observation_from_spacial_info_with_noise,
                    next_closest_index=additional_info.next_closest_index,
                    reward=additional_info.reward,
                )
            ]
        
        
        if config.action_adjuster.use_threading:
            sleep(time_slowdown)
        
        # 
        # receive data from solver thread
        # 
        if self.should_update_solution():
            print(f'''main thread:len(shared_thread_data["timestep_data"]) = {len(shared_thread_data["timestep_data"])}''')
            # if not multithreading: run solver manually
            if not config.action_adjuster.use_threading:
                self.solver.fit_points()
            
            # pull in any records that need logging
            with print.indent: 
                self.records_from_the_solver_thread += shared_thread_data["records_to_log"]
                shared_thread_data["records_to_log"] = []
            
                # pull in a new solution
                new_solution = shared_thread_data.get("transform_json", None)
                if new_solution:
                    self.active_transform = Transform.from_json(
                        json.loads(
                            new_solution
                        )
                    )
    def when_episode_ends(self):
        pass
    
    def when_mission_ends(self):
        pass


# this little thing is because python multithreading is stupid and needs to have Manager() created in the main file
# this is just a hacky way around that
@bb.run_in_main
def _():
    global shared_thread_data
    shared_thread_data = Manager().dict() if config.action_adjuster.use_threading else {}
        
bb.run_main_hooks_if_needed(__name__)