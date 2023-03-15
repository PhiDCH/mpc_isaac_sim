import rospy 
from geometry_msgs.msg import Twist
import tf
import numpy as np 
import time
import threading as th
from matplotlib import pyplot as plt 
from matplotlib.patches import Ellipse
from util import euler_from_quaternion
from mpc_solver import MPCController, rob_diam
from scan import CostMapConverter


 

TARGET_POINT = np.array([8.0, 0.0, 0.0])    # x,y
# TARGET_POINT = np.array([0.0, 0.0, 0.0])    # x,y

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
        
        self.limit = 2.0
        self.lidar = CostMapConverter(self.limit)
        self.lidar.start_sub_lidar(topic_name='/carter1/scan')

        self.mpc = MPCController()

    def get_state_from_tf(self,):
        trans,rot = self.listener.lookupTransform("world", "carter1/base_link", rospy.Time(0))
        rot = euler_from_quaternion(rot)
        return np.array([trans[0],trans[1], rot])

    def run(self,)->None:
        fig, ax = plt.subplots()

        rate = rospy.Rate(10.0)
        th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
        while not rospy.is_shutdown():
            ax.set(xlim=(-self.limit,self.limit), ylim=(-self.limit,self.limit),
            xticks=np.arange(-self.limit,self.limit), yticks=np.arange(-self.limit,self.limit))

            ells = self.lidar.get_obstackle()
            for item in ells:
                ell, xy = item
                ax.scatter(xy[:,0], xy[:,1], s=0.1)
                x,y,a,b,rot = ell
                ell = Ellipse((x,y), width=a, height=b, angle=rot, fill=False)
                ax.add_patch(ell)
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
                ax.plot(-X0[1], X0[0])
                # print(np.around(u, 4))

            else:
                self.set_cmd_vel(0.0, 0.0)
                print('done')

            cir = plt.Circle((0.0,0.0), rob_diam, fill=False)
            ax.add_artist(cir)
            cir1 = plt.Circle((-pose_target[1], pose_target[0]), 0.2, fill=True)
            ax.add_artist(cir1)
            rate.sleep()
            plt.pause(0.00001)
            ax.cla()

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
