from copy import deepcopy
import itertools
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion as qut
import message_filters
from __dependencies__.blissful_basics import print, LazyDict, stringify

from generic_tools.universe.timestep import Timestep
from config import config, path_to

debug = False
config.ros_runtime.is_client = True

# 
# client
# 
this_file = __file__ # for some reason, __file__ changes throughout the runtime, so the value needs to be saved in another variable
class RosRuntime:
    # needs env just for recording data, the env itself is basically disabled
    def __init__(self, agent, env):
        self.agent = agent
        self.env   = env
        self.is_first_observation = True
        self.previous_action      = None
        
        self.agent.when_mission_starts()
        
        self.agent.previous_timestep = Timestep(
            index=-2,
        )
        self.agent.timestep = Timestep(
            index=-1,
        )
        self.agent.next_timestep = Timestep(
            index=0,
            observation=None, # would normally be env.reset(), but must wait for GPS
            is_last_step=False,
        )
        self.first_observation_loaded = False
        
        rospy.init_node(config.ros_runtime.main_node_name)
        self.controller_publisher = rospy.Publisher(
            rospy.get_param("~cmd_topic", config.ros_runtime.controller_topic),
            Twist,
            queue_size=20,
        )
        self.odom_subscriber = message_filters.Subscriber(
            rospy.get_param('~odom_topic', config.ros_runtime.odometry_topic),
            Odometry,
        )
        self.time_synchonizer = message_filters.ApproximateTimeSynchronizer(
            [
                self.odom_subscriber,
            ],
            config.ros_runtime.time_sync_size,
            1,
            allow_headerless=True
        )
        self.time_synchonizer.registerCallback(self.when_data_arrives)
        if config.ros_faker.enable:
            import subprocess
            from blissful_basics import FS
            import sys
            debugging = True
            _process = subprocess.Popen(
                [
                    sys.executable,
                    this_file,
                ],
                **(dict(stdout=sys.stdout) if debugging else dict(stdout=subprocess.PIPE)),
                # stderr=subprocess.STDOUT,
            )
        debug and print("waiting for odom message")
        rospy.spin()
    
    def publish_action(self, action):
        velocity, spin = action
        if not rospy.is_shutdown():
            message = Twist()
            message.linear.x = velocity
            message.angular.z = spin
            debug and print("publishing control")
            self.controller_publisher.publish(message)
            debug and print("published control")
    
    def when_data_arrives(self, odom_msg):
        debug and print(f'''got odom_msg''')
        x        = odom_msg.pose.pose.position.x
        y        = odom_msg.pose.pose.position.y
        angle    = odom_msg.pose.pose.position.z
        velocity = odom_msg.twist.twist.linear.x
        spin     = odom_msg.twist.twist.angular.x
        
        # velocity = odom_msg.twist.twist.linear.x 
        # spin     = odom_msg.twist.twist.angular.z 
        # x        = odom_msg.pose.pose.position.x
        # y        = odom_msg.pose.pose.position.y
        # temp_y   = odom_msg.pose.pose.orientation.z
        # temp_x   = odom_msg.pose.pose.orientation.w
        # angle = qut((temp_x, 0, 0, temp_y)).radians*numpy.sign(temp_y)
        
        new_spacial_info = self.env.SpacialInformation( x, y, angle, velocity, spin, math.inf )
        
        env   = self.env
        agent = self.agent
        
        # initalize on first run
        if not self.first_observation_loaded:
            agent.next_timestep.observation = env.reset(override_next_spacial_info=new_spacial_info)
            debug and config.ros_runtime.is_client and print(f'''first_observation = {stringify(agent.next_timestep.observation)}''')
            agent.when_episode_starts()
        
            self.first_observation_loaded = True
        else:
            # previous reaction was based on previous observation, but we can't call env.step until after we have the NEXT observation (e.g. this func call is that "NEXT")
            observation, reward, is_last_step, agent.timestep.hidden_info = env.step(self.previous_action, override_next_spacial_info=new_spacial_info)
            agent.next_timestep.observation = deepcopy(observation)
            agent.timestep.reward           = deepcopy(reward)
            agent.timestep.is_last_step     = deepcopy(is_last_step)
            agent.when_timestep_ends()
        
        agent.previous_timestep = agent.timestep
        agent.timestep          = agent.next_timestep
        agent.next_timestep     = Timestep(index=agent.next_timestep.index+1)
        
        # always compute the reaction to the data that arrived
        agent.when_timestep_starts()
        reaction = agent.timestep.reaction
        debug and config.ros_runtime.is_client and print(f'''reaction = {reaction}''')
        if type(reaction) == type(None):
            reaction = env.action_space.sample()
        self.publish_action(reaction)
        self.previous_action = reaction
        
    def __del__(self):
        self.agent.when_episode_ends()
        self.agent.when_mission_ends()

# 
# warthog ros fake server 
# 
if __name__ == "__main__":
    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    import message_filters

    from statistics import mean as average
    from random import random, sample, choices

    import torch
    from rigorous_recorder import RecordKeeper
    from stable_baselines3 import PPO
    from __dependencies__.blissful_basics import FS, print, LazyDict

    from envs.warthog import WarthogEnv
    from config import config, path_to

    recorder = RecordKeeper(config=config)
    config.ros_runtime.is_client = False
    with print.indent:
        # 
        # env
        # 
        env = WarthogEnv(
            waypoint_file_path=path_to.default_waypoints,
            trajectory_output_path=f"{config.output_folder}/trajectory.log",
            recorder=recorder,
        )


        # 
        # setup ROS
        # 
        rospy.init_node(config.ros_faker.main_node_name)
        controller_topic = rospy.get_param("~cmd_topic", config.ros_runtime.controller_topic)
        odometry_topic   = rospy.get_param('~odom_topic', config.ros_runtime.odometry_topic)
        controller_subscriber = message_filters.Subscriber(
            controller_topic,
            Twist,
        )
        odom_publisher = rospy.Publisher(
            odometry_topic,
            Odometry,
            queue_size=20,
        )

        # 
        # send out messages/responses 
        # 
        def publish_position():
            odom_msg = Odometry()
            odom_msg.pose.pose.position.x  = env.spacial_info.x
            odom_msg.pose.pose.position.y  = env.spacial_info.y
            odom_msg.pose.pose.position.z  = env.spacial_info.angle
            odom_msg.twist.twist.linear.x  = env.spacial_info.velocity
            odom_msg.twist.twist.angular.x = env.spacial_info.spin
            # print(f'''server: env.spacial_info = {env.spacial_info}''')
            debug and print("publishing odom message")
            odom_publisher.publish(odom_msg)
            debug and print("published odom message")

        @controller_subscriber.registerCallback
        def when_controller_command_sent(message):
            debug and print(f'''received control = {message}''')
            velocity = message.linear.x
            spin     = message.angular.z   
            action = [ velocity, spin ]
            # print(f'''server: action = {action}''')
            env.step(action)
            publish_position()

        env.reset()
        import time
        time.sleep(1) # NOTE: this is VERY important. I have no idea why, but things just do not work if this sleep is not here
        publish_position()
        rospy.spin()