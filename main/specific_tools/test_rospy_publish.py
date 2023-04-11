import rospy
from geometry_msgs.msg import Twist
import message_filters
from main.config import config, path_to
from time import sleep

rospy.init_node(config.ros_runtime.main_node_name)
controller_topic = rospy.get_param("~cmd_topic", config.ros_runtime.controller_topic)
controller_publisher = rospy.Publisher(
    controller_topic,
    Twist,
    queue_size=1,
)
while True:
    sleep(1)
    message = Twist()
    print("publishing")
    controller_publisher.publish(message)