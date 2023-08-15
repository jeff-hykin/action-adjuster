import rospy
from geometry_msgs.msg import Twist
import message_filters
from config import config, path_to
import atexit
from blissful_basics import Warnings
from nav_msgs.msg import Odometry

# Warnings.disable()

rospy.init_node(config.ros_faker.main_node_name+"2")

# 
# controller Subscriber
#
if False:
    controller_topic = rospy.get_param("~cmd_topic", config.ros_runtime.controlleretry_topic)
    print(f'''controller_topic = {controller_topic}''')
    def callback(*args):
        print(f'''args = {args}''')
        
    controller_subscriber = rospy.Subscriber(
        controller_topic,
        Odometry,
        callback,
    )

# 
# odom Subscriber
#
if True:
    odom_topic = rospy.get_param("~cmd_topic", config.ros_runtime.odometry_topic)
    print(f'''odom_topic = {odom_topic}''')
    def callback(*args):
        print(f'''args = {args}''')
        
    odom_subscriber = rospy.Subscriber(
        odom_topic,
        Odometry,
        callback,
    )

rospy.spin()