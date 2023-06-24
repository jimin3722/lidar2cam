#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing

# External modules  
import cv2
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from lidar2cam import LIDAR2CAMTransform

# ROS modules
PKG = 'lidar_camera_calibration'
import roslib; roslib.load_manifest(PKG)
import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_matrix
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2

# Global variables
TF_BUFFER = None
CV_BRIDGE = CvBridge()

# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'

LC = LIDAR2CAMTransform(1280, 720, 44.5)

class lidar2cam():
    def __init__(self):
        self.img_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback, queue_size = 1)
        self.lidar_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.lidar_callback, queue_size = 1)
        self.lidar_input_flag = False
    
    def img_callback(self, img_msg):
        self.img_msg = img_msg

    def lidar_callback(self, velodyne):
        print("asdf")
        self.velodyne = velodyne
        self.lidar_input_flag = True

    def project_point_cloud(self):
        try:
            img = CV_BRIDGE.imgmsg_to_cv2(self.img_msg, 'bgr8')
        except CvBridgeError as e: 
            rospy.logerr(e)
            return
        try:
            transform = TF_BUFFER.lookup_transform('world', 'velodyne', rospy.Time())
            velodyne = do_transform_cloud(self.velodyne, transform)
        except tf2_ros.LookupException:
            pass

        points = pc2.read_points_list(velodyne,field_names = ("x", "y", "z"), skip_nans=True)
        objPoints = np.empty((1, 3))
        for p in points:
            objPoints = np.append(objPoints, [[p[0], p[1], p[2]]], axis = 0)
        points = objPoints.tolist()
        points3D = np.asarray(objPoints.tolist())
        
        # Filter points in front of camera
        inrange = np.where((points3D[:, 2] > -2) &
                        (points3D[:, 2] < 6) &
                        (points3D[:, 0] < 5) &
                        (points3D[:, 0] > 0.1) &
                        (np.abs(points3D[:, 1]) < 4))
        max_intensity = np.max(points3D[:, -1])
        points3D = points3D[inrange[0]]

        xyc = LC.transform_lidar2cam(points3D)
        xyi = LC.project_pts2ing(xyc) 
    
        for i in range(len(xyi)):
            cv2.circle(img, tuple(xyi[i]), 2, (255,255,255), -1)
            # cv2.resize(img,(480,640))
        try:
            cv2.imwrite("./target.png",img)
            print("is:",img.shape)
            cv2.imshow('sss' ,img)
            cv2.waitKey(0)
        except CvBridgeError as e: 
            rospy.logerr(e)

if __name__ == '__main__':

    lc = lidar2cam()

    while not rospy.is_shutdown():
        if lc.lidar_input_flag == True:
            print("sasd")
            lc.project_point_cloud()
            lc.lidar_input_flag = False