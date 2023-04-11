import rospy
from geometry_msgs.msg import Twist
import message_filters
from main.config import config, path_to

rospy.init_node(config.ros_faker.main_node_name)
controller_topic = rospy.get_param("~cmd_topic", config.ros_runtime.controller_topic)
controller_subscriber = message_filters.Subscriber(
    controller_topic,
    Twist,
)
@controller_subscriber.registerCallback
def callback(*args):
    print(f'''args = {args}''')

rospy.spin()