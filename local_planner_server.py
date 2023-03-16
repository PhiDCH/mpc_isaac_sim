import rospy 
from geometry_msgs.msg import Twist
import tf
import numpy as np 
import time
import threading as th
from matplotlib.patches import Ellipse
from util import euler_from_quaternion
from mpc_solver import MPCController, rob_diam
from scan import CostMapConverter
from map import draw_map, add_tf

 

TARGET_POINT = np.array([8.0, 0.0, 0.0])    # x,y
# TARGET_POINT = np.array([0.0, 0.0, 0.0])    # x,y


class IsaacSim():
    def __init__(self,):
        self.listener = tf.TransformListener()
        self.listener.waitForTransform("world", "carter1/base_link", rospy.Time(), rospy.Duration(4.0))

        self.cmd_pub = rospy.Publisher('carter1/cmd_vel', Twist, queue_size=1)
        
        self.limit = 2.0
        self.lidar = CostMapConverter(self.limit)
        self.lidar.start_sub_lidar(topic_name='/carter1/scan')

        self.mpc = MPCController()

    def get_state_from_tf(self,):
        trans,rot = self.listener.lookupTransform("world", "carter1/base_link", rospy.Time(0))
        rot = euler_from_quaternion(rot)
        return np.array([trans[0],trans[1], rot])

    def run(self,)->None:
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            ells = self.lidar.get_obstackle()
            for item in ells:
                ell, xy = item
                x,y,a,b,rot = ell
                ell = Ellipse((x,y), width=a, height=b, angle=rot, fill=False)
            ells_ = [ell[0] for ell in ells]    
            
            pose_init = self.get_state_from_tf()
            # print(np.around(pose_init, 4))
            pose_target = TARGET_POINT - pose_init
            # print(np.around(pose_target, 4))
            error = np.linalg.norm(pose_target[:2])
            if error > 0.1:

                u, X0 = self.mpc.step(np.array([0,0,pose_init[2]]), pose_target, ells_)
                # u, X0 = self.mpc.step(np.array([0,0,pose_init[2]]), pose_target)
                u = np.array(u.full()).T
                u0 = u[0]
                # print(u0)
                self.set_cmd_vel(u0[0], u0[1])
                X0 = np.array(X0.full())
                # print(np.around(u, 4))

            else:
                self.set_cmd_vel(0.0, 0.0)
                print('done')

        self.lidar.release_sub_lidar()
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
