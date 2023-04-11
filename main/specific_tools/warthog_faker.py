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
    print("publishing odom message")
    odom_publisher.publish(odom_msg)
    print("published odom message")

@controller_subscriber.registerCallback
def when_controller_command_sent(message):
    print(f'''received control = {message}''')
    velocity = message.linear.x
    spin     = message.angular.z   
    action = [ velocity, spin ]
    env.step(action)
    publish_position()

# env.reset()
from time import sleep
sleep(1) # PAIN: this sleep is VERY important, I have no idea why but removing it breaks the ros events and freezes

publish_position()
rospy.spin()