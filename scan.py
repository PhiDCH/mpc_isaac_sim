import rospy
from sensor_msgs.msg import LaserScan
import numpy as np 
import time
from util import polar2decart, cluster, fit2ellipse
from matplotlib import pyplot as plt 
from matplotlib.patches import Ellipse
import cv2
import threading as th

plt.ion()
keep_going = True
def key_capture_thread():
    global keep_going
    input()
    keep_going = False


class CostMapConverter():
    def __init__(self,):
        self.sub_lidar = None
        self.scan = None

    def start_sub_lidar(self, topic_name: str = '/scan') -> bool:
        self.sub_lidar = rospy.Subscriber(
            topic_name, LaserScan, self.lidar_callback)
        rate = rospy.Rate(10)
        t1 = time.time()
        # wait 10s until subscrube done
        while not rospy.is_shutdown() and (time.time()-t1) < 3:
            if self.scan is not None:
                rospy.loginfo(f'subscribe {topic_name} done')
                return True
            rate.sleep()
        rospy.loginfo(f'fail to subscribe {topic_name}')
        return False

    def lidar_callback(self, msg: LaserScan)->None:
        self.scan = np.array(msg.ranges) 
        return

    def release_sub_lidar(self,) -> None:
        if self.sub_lidar:
            self.sub_lidar.unregister()
        return

    def run(self,) -> None:
        rate = rospy.Rate(10)
        if not self.start_sub_lidar(topic_name='/carter1/scan'):
            return
        # print(self.scan.shape)
        # return
        fig, ax = plt.subplots()
        limit = 15.0
        # ax.set(xlim=(-limit,limit), ylim=(-limit,limit),
        #     xticks=np.arange(-limit,limit), yticks=np.arange(-limit,limit))

        # press 'Enter' to break 
        th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
        while not rospy.is_shutdown() and keep_going:
            ax.set(xlim=(-limit,limit), ylim=(-limit,limit),
            xticks=np.arange(-limit,limit), yticks=np.arange(-limit,limit))
            
            xy = polar2decart(self.scan, limit=limit)
            if len(xy) > 0:
                clus = cluster(xy)
                for i in set(clus):
                    if i>-1:
                        xy_ = xy[np.where(clus==i)]
                        ax.scatter(xy_[:,0], xy_[:,1])
                        x,y,a,b,rot = fit2ellipse(xy_, n_std=1.0)
                        ell = Ellipse((x,y), width=2*a, height=2*b, angle=rot, fill=False)
                        # print(rot)
                        ax.add_patch(ell)
                        # np.save('d1.npy', xy_)
                        # break

            rate.sleep()
            plt.pause(0.00001)

            ax.cla()

        self.release_sub_lidar()
        print('stop run')
        return


if __name__=='__main__':
    rospy.init_node('test_lidar_node')
    cmv = CostMapConverter()
    cmv.run()
    rospy.spin()


