from envs.warthog import WarthogEnv
from config import config, path_to

env = WarthogEnv(
    waypoint_file_path=path_to.default_waypoints,
    trajectory_output_path="logs.ignore/trajectory.log"
)
print(f'''env.reset() = {env.reset()}''')
print(f'''env.step([0,0]) = {env.step([0,0])}''')
print(f'''env.step([0,0]) = {env.step([0,0])}''')