import rospy
from geometry_msgs.msg import Twist
import message_filters
from config import config, path_to
from time import sleep


# 
# controller publish
# 
if False:
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
        print("publishing controller message")
        controller_publisher.publish(message)

# 
# odom publish
#
if True:
    from nav_msgs.msg import Odometry
    rospy.init_node(config.ros_faker.main_node_name)
    odometry_topic   = rospy.get_param('~odom_topic', config.ros_runtime.odometry_topic)
    odom_publisher = rospy.Publisher(
        odometry_topic,
        Odometry,
        queue_size=1,
    )
    while True:
        sleep(1)
        message = Odometry()
        print("publishing odom message")
        odom_publisher.publish(message)