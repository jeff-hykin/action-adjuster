from copy import deepcopy
import itertools
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion as qut
import message_filters
from blissful_basics import print, LazyDict

from generic_tools.universe.timestep import Timestep
from config import config, path_to

class RosRuntime:
    def __init__(self, agent, env):
        self.agent = agent
        self.env   = env
        self.is_first_observation = True
        self.previous_action      = None
        self.timestep_count       = 0
        
        self.agent.when_mission_starts()
        self.timestep_count = 0
        
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
            queue_size=1,
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
                    FS.local_path("../specific_tools/warthog_faker.py",),
                ],
                **(dict(stdout=sys.stdout) if debugging else dict(stdout=subprocess.PIPE)),
                # stderr=subprocess.STDOUT,
            )
        print("waiting for odom message")
        rospy.spin()
    
    def publish_action(self, action):
        print("publishing control")
        velocity, spin = action
        if not rospy.is_shutdown():
            message = Twist()
            message.linear.x = velocity
            message.angular.z = spin
            self.controller_publisher.publish(message)
    
    def when_data_arrives(self, odom_msg):
        print(f'''got odom_msg''')
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
        
        new_spacial_info = self.env.SpacialInformation([ x, y, angle, velocity, spin ])
        
        env   = self.env
        agent = self.agent
        
        # initalize
        if not self.first_observation_loaded:
            agent.next_timestep.observation = env.reset(spacial_info_override=new_spacial_info)
            self.first_observation_loaded = True
            agent.when_episode_starts()
        
        
        self.timestep_count += 1
        
        agent.previous_timestep = agent.timestep
        agent.timestep          = agent.next_timestep
        agent.next_timestep     = Timestep(index=agent.next_timestep.index+1)
        
        agent.when_timestep_starts()
        reaction = agent.timestep.reaction
        if type(reaction) == type(None):
            reaction = env.action_space.sample()
        self.publish_action(reaction)
        observation, reward, is_last_step, agent.timestep.hidden_info = env.step(reaction, spacial_info_override=new_spacial_info)
        agent.next_timestep.observation = deepcopy(observation)
        agent.timestep.reward           = deepcopy(reward)
        agent.timestep.is_last_step     = deepcopy(is_last_step)
        agent.when_timestep_ends()
    
    def __del__(self):
        self.agent.when_episode_ends()
        self.agent.when_mission_ends()
    