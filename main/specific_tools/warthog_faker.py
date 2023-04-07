import rospy
from geometry_msgs.msg import Twist

def main():
    rospy.init_node("command_publisher_node")
    cmd_vel_topic = rospy.get_param("~cmd_topic", "/warthog_velocity_controller/cmd_vel")
    cmd_vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size = 1)
    start_v = 1.
    start_w = -4.
    vel_samples = 5
    w_samples = 40
    for i in range(0,5 - int(start_v)):
        start_w = -4.
        r = rospy.Rate(1)
        for o in range(0,50):
            r.sleep()
        for j in range(0,40):
            t = 0 
            start_time = rospy.get_rostime()
            r = rospy.Rate(15)
            while t < 6 and not rospy.is_shutdown():
                cmd_vel = Twist()
                cmd_vel.linear.x = start_v
                cmd_vel.angular.z = start_w
                cmd_vel_pub.publish(cmd_vel)
                r.sleep()
                t = (rospy.get_rostime() - start_time).to_sec()
                #print(t)
            start_w = start_w + 0.2
            if rospy.is_shutdown():
                break
        
        if rospy.is_shutdown():
            break
        break
        start_v = start_v + 1

if __name__=="__main__":
    main()
