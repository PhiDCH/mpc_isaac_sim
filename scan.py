import rospy
from sensor_msgs.msg import LaserScan
import numpy as np 

data = []
def callback(msg):
    # print(len(msg.ranges))
    # print('angle min', msg.angle_min)
    # print('angle max', msg.angle_max)
    # print('angle_increment', msg.angle_increment)
    global data
    data = msg.ranges
    print(len(data))
    np.save('data.npy', np.array(data, dtype=np.float32), allow_pickle=True)

rospy.init_node('scan_values')
sub = rospy.Subscriber('/scan', LaserScan, callback)
while not rospy.is_shutdown():
    if data:
        sub.unregister()
        print('break')
        break
    rospy.spin()

print(len(data))

