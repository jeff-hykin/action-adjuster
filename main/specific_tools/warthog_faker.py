import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import message_filters

from statistics import mean as average
from random import random, sample, choices

import torch
from rigorous_recorder import RecordKeeper
from stable_baselines3 import PPO
from blissful_basics import FS, print, LazyDict

from envs.warthog import WarthogEnv
from config import config, path_to

recorder = RecordKeeper(config=config)

# 
# env
# 
env = WarthogEnv(
    waypoint_file_path=path_to.default_waypoints,
    trajectory_output_path=f"{path_to.default_output_folder}/trajectory.log",
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
    odometry_topic
    Odometry,
    queue_size=1,
)
time_synchonizer = message_filters.ApproximateTimeSynchronizer(
    [
        controller_subscriber,
    ],
    config.ros_runtime.time_sync_size,
    1,
    allow_headerless=True
)


# 
# send out messages/responses 
# 
def publish_position():
    odom_msg = Odometry()
    odom_msg.twist.twist.linear.x       = env.spacial_info.x 
    odom_msg.twist.twist.angular.z      = env.spacial_info.y     
    odom_msg.twist.twist.position.x     = env.spacial_info.angle        
    odom_msg.twist.twist.position.y     = env.spacial_info.velocity        
    odom_msg.twist.twist.orientation.z  = env.spacial_info.spin    # angle = qut((temp_x, 0, 0, temp_y)).radians*numpy.sign(temp_y)
    # odom_msg.twist.twist.orientation.w  = temp_x   
    odom_publisher.publish(odom_msg)

env.reset()
publish_position()
def when_controller_command_sent(message):
    print(f'''command = {message}''')
    velocity = message.linear.x
    spin     = message.angular.z   
    action = [ velocity, spin ]
    env.step(action)
    publish_position()

time_synchonizer.registerCallback(when_controller_command_sent)