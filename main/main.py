from envs.warthog import WarthogEnv
from prediction import project, prediction_loss, ActionAdjuster
from config import config, path_to
import torch

# FIXME: need a real policy
def policy(observation):
    velocity_action = 0.5
    spin_action     = 0.01
    return velocity_action, spin_action

action_adjuster = ActionAdjuster(policy=policy, initial_transform=[1,1])

env = WarthogEnv(
    waypoint_file_path=path_to.default_waypoints,
    trajectory_output_path="logs.ignore/trajectory.log"
)

observation = env.reset()
while True:
    action          = policy(observation)
    adjusted_action = action_adjuster.adjust(action)
    observation, reward, done, additional_info = env.step(adjusted_action)
    action_adjuster.add_data(observation, additional_info)
    if done:
        break