#!/usr/bin/env python
# -- coding: utf-8 --
import rospy
import time
import math
import numpy as np
from collections import defaultdict
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud, ChannelFloat32, Imu, Image


from con_classifier import Cone_Classfier
from cv_bridge import CvBridge, CvBridgeError
cc = Cone_Classfier()
CV_BRIDGE = CvBridge()

class Cone_detection:
    def __init__(self):
        self.lidar_sub=rospy.Subscriber('object3D',PointCloud,self.callback,queue_size=1)
        self.cone_pub=rospy.Publisher('cone',PointCloud,queue_size=1)


        self.img_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback, queue_size = 1)
        self.img_msg = 0

        self.cone_max_lenth=0.4 #m단위 콘 폭 0.37 높이 0.7
        self.cone_max_height=0.7

    
    def img_callback(self, img_msg):
        self.img_msg = img_msg


    def callback(self, input_rosmsg):
        label = []
        point = [[p.x, p.y, p.z] for p in input_rosmsg.points]
        #print(point)
        
        for channel in input_rosmsg.channels:        
            label = [c for c in channel.values]
        
        label_points = defaultdict(list)
        for l, p in zip(label, point):
            label_points[l].append(p)

        cone_centers=[]
        #print(label_points)
        for i in label_points:
            cone_points=label_points.get(i)
            x_list=[]
            y_list=[]
            z_list=[]
            for k in cone_points:
                x_list.append(k[0])
                y_list.append(k[1])
                z_list.append(k[2])
            x_range=max(x_list)-min(x_list)
            y_range=max(y_list)-min(y_list)
            z_range=max(z_list)-min(z_list)
            
            if x_range>0.05 and x_range<0.55 and y_range>0.05 and y_range<0.55 and z_range>0.05 and z_range<0.85:
                x_mean=sum(x_list)/len(x_list)
                y_mean=sum(y_list)/len(y_list)
                z_mean=sum(z_list)/len(z_list)
                cone_centers.append([x_mean,y_mean,z_mean])
            elif max(x_list)<3 and x_range>0.05 and x_range<0.55 and y_range>0.05 and y_range<0.55 and z_range<x_range/4 and z_range>0.05:
                x_mean=sum(x_list)/len(x_list)
                y_mean=sum(y_list)/len(y_list)
                z_mean=sum(z_list)/len(z_list)
                cone_centers.append([x_mean,y_mean,z_mean])   

        print(cone_centers)
        try:
            img = CV_BRIDGE.imgmsg_to_cv2(self.img_msg, 'bgr8')
            # img = np.frombuffer(self.img_msg.data, dtype=np.uint8).reshape(self.img_msg.height, self.img_msg.width, -1)
        except CvBridgeError as e: 
            rospy.logerr(e)
            return        
        classfied_cone = cc.cone_classifier(cone_centers, img)


        self.cones = PointCloud()
        self.cones.header.frame_id='map'
        for i in cone_centers:
            point=Point32()
            point.x=i[0]
            point.y=i[1]
            point.z=0
            self.cones.points.append(point)

        self.cone_pub.publish(self.cones)
        print("콘의 개수: ",len(cone_centers))
        


def main():
    rospy.init_node('cone_detection',anonymous=True)
    Cone=Cone_detection()
    while not rospy.is_shutdown():
        pass

if __name__ == '__main__':
    main()