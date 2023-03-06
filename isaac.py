import rospy 
from geometry_msgs.msg import Twist
import tf
import numpy as np 
import time
import threading as th
from matplotlib import pyplot as plt 
import math
 
def euler_from_quaternion(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x,y,z,w = quat
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        # yaw_z = np.rad2deg(yaw_z)
        return yaw_z


TARGET_POINT = np.array([-4.0, 2.0])    # x,y

plt.ion()
keep_going = True
def key_capture_thread():
    global keep_going
    input()
    keep_going = False

class IsaacSim():
    def __init__(self,):
        self.listener = tf.TransformListener()
        self.listener.waitForTransform("world", "carter1/base_link", rospy.Time(), rospy.Duration(4.0))

        self.cmd_pub = rospy.Publisher('carter1/cmd_vel', Twist, queue_size=1)

    def get_state_from_tf(self,):
        trans,rot = self.listener.lookupTransform("world", "carter1/base_link", rospy.Time(0))
        rot = euler_from_quaternion(rot)
        return np.array([trans[0],trans[1], rot])

    def run(self,)->None:
        print(self.get_state_from_tf())
        # rate = rospy.Rate(10.0)
        # while not rospy.is_shutdown():
        #     (trans,rot) = self.listener.lookupTransform("world", "carter1/base_link", rospy.Time(0))
        #     # print(np.around(euler_from_quaternion(rot), 4))
        #     print(np.around(trans, 4))
        #     self.set_cmd_vel(0.5, 0.0)
        #     rate.sleep()
        return

    def set_cmd_vel(self, vel_x, vel_yaw):
        twist = Twist()
        twist.linear.x = vel_x
        twist.angular.z = vel_yaw
        self.cmd_pub.publish(twist)

    

if __name__=='__main__':
    rospy.init_node('test_isaac')
    sim = IsaacSim()
    sim.run()
    rospy.spin()