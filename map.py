#!/usr/bin/python3

import rospy
import cv2 
import numpy as np


MAP_IMAGE = 'hospital.png'
MAP_META = 'hospital.yaml'
NODE_LIST = 'coordinate_of_nodes.xlsx'


from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import ColorRGBA
import pandas as pd 

def draw_map():
    ### publish static map
    img = cv2.imread(MAP_IMAGE, cv2.IMREAD_GRAYSCALE)
    img = cv2.flip(img, 0)
    h,w = img.shape
    img = np.int8(80*(img/255)).flatten()

    pub = rospy.Publisher('map', OccupancyGrid, latch=True, queue_size=1)

    map_msg = OccupancyGrid()
    map_msg.data = img
    map_msg.info.width = w
    map_msg.info.height = h

    map_msg.info.resolution = 0.05
    map_msg.info.origin.position = Point(-26.925000762939455, -37.07500228881836, 0)       # bottom left


    pub.publish(map_msg)

    ### publish markers
    listnode = pd.read_excel(NODE_LIST)

    pub1 = rospy.Publisher('marker', MarkerArray, latch=True, queue_size=10)
    markers = []

    for i in range(len(listnode)):
        marker = Marker()
        marker.action = marker.ADD    
        marker.header.frame_id = "map"
        marker.ns = 'node'
        marker.id = 2*i
        marker.type = marker.CYLINDER
        marker.pose.position = Point(-listnode.X[i], -listnode.Y[i],0)
        marker.pose.orientation = Quaternion(0,0,0,1)
        marker.scale = Vector3(1.5,1.5,0.01)
        marker.color = ColorRGBA(0, 1, 0, 0.2)

        markers.append(marker)

        marker = Marker()
        marker.id = 2*i+1
        marker.action = marker.ADD
        marker.header.frame_id = "map"
        marker.type = Marker.TEXT_VIEW_FACING
        marker.text = listnode.Point[i]
        marker.color = ColorRGBA(1, 0, 0, 0.8)
        marker.pose.position = Point(-listnode.X[i], -listnode.Y[i],0)
        marker.scale = Vector3(0,0,0.8)

        markers.append(marker)

    pub1.publish(markers)


import tf
from util import euler_from_quaternion


def add_tf():
    listener = tf.TransformListener()
    listener.waitForTransform("world", "carter1/base_link", rospy.Time(), rospy.Duration(4.0))
    
    br = tf.TransformBroadcaster()
    br1 = tf.TransformBroadcaster()
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        trans,rot = listener.lookupTransform("world", "carter1/base_link", rospy.Time(0))
        x = -trans[0]
        y = -trans[1]
        # br1.sendTransform((0,0,0), (0,0,0,1), rospy.Time.now(), "world", "map")
        br.sendTransform((x,y,0), rot, rospy.Time.now(), "carter1", "map")
        rate.sleep()


if __name__=="__main__":
    rospy.init_node('map')
    draw_map()
    add_tf()
    rospy.spin()