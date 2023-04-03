from copy import deepcopy
import itertools
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion as qut
from vn300.msg import ins
from pacmod_msgs.msg import SystemRptFloat
import message_filters
from blissful_basics import print, LazyDict

from generic_tools.universe.timestep import Timestep

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
        
        # FIXME: setup ros
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', 'warthog_velocity_controller/cmd_vel')
        #self.odom_topic = rospy.get_param('~odom_topic', 'odometry/filtered_map')
        self.odom_topic = rospy.get_param('~odom_topic', 'odometry/filtered')
        self.gps_odom_topic = rospy.get_param('~map_odom_topic', 'odometry/filtered2')
        #self.odom_topic = rospy.get_param('~odom_topic', 'warthog_velocity_controller/odom')
        # self.out_file = rospy.get_param('~out_file_name', 'real_remote_poses_ext_war_gps2.csv')
        self.out_file = rospy.get_param('~out_file_name', 'rellis_mud1.csv')
        self.file_h = open(self.out_file, 'w')
        self.file_h.writelines("x,y,th,vel,w,v_cmd,w_cmd\n")
        self.cmd_vel_sub = message_filters.Subscriber(self.cmd_vel_topic, Twist)
        self.odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)
        self.gps_odom_sub = message_filters.Subscriber(self.gps_odom_topic, Odometry)
    
    def publish_action(self, action):
        # FIXME
        pass
    
    def when_data_arrives(self, odom_msg, gps_msg):
        velocity = odom_msg.twist.twist.linear.x 
        spin     = odom_msg.twist.twist.angular.z 
        x      = gps_msg.pose.pose.position.x
        y      = gps_msg.pose.pose.position.y
        temp_y = gps_msg.pose.pose.orientation.z
        temp_x = gps_msg.pose.pose.orientation.w
        angle = qut((temp_x, 0, 0, temp_y)).radians*numpy.sign(temp_y)
        
        new_spacial_info = env.SpacialInformation([ x, y, angle, velocity, spin ])
        
        env   = self.env
        agent = self.agent
        
        # initalize
        if not self.first_observation_loaded:
            agent.next_timestep.observation = env.reset(spacial_info_override=new_spacial_info)
            self.first_observation_loaded = True
            agent.when_episode_starts()
        # typical timestep incrementation
        else:
            self.timestep_count += 1
            
            agent.previous_timestep = agent.timestep
            agent.timestep          = agent.next_timestep
            agent.next_timestep     = Timestep(index=agent.next_timestep.index+1)
            
            agent.when_timestep_starts()
            reaction = agent.timestep.reaction
            if type(reaction) == type(None):
                reaction = env.action_space.sample()
            observation, reward, is_last_step, agent.timestep.hidden_info = env.step(reaction, spacial_info_override=new_spacial_info)
            agent.next_timestep.observation = deepcopy(observation)
            agent.timestep.reward           = deepcopy(reward)
            agent.timestep.is_last_step     = deepcopy(is_last_step)
            agent.when_timestep_ends()
    
    def __del__(self):
        self.agent.when_episode_ends()
        self.agent.when_mission_ends()
    