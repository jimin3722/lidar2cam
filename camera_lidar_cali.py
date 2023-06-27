#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing

# External modules  
import cv2
import numpy as np

from lidar2cam import LIDAR2CAMTransform
import lidar_module

import rospy
from cv_bridge import CvBridge, CvBridgeError
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2

# Global variables
TF_BUFFER = None
CV_BRIDGE = CvBridge()

LC = LIDAR2CAMTransform(1280, 720, 44.5)

class lidar2cam():
    def __init__(self):
        self.img_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback, queue_size = 1)
        self.lidar_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.lidar_callback, queue_size = 1)
        self.lidar_input_flag = False
        self.velodyne = 0
        self.img_msg = 0
    
    def img_callback(self, img_msg):
        self.img_msg = img_msg

    def lidar_callback(self, velodyne):
        print("----------lidar_come------------")
        # self.velodyne = velodyne
        self.obs_xyz=list(map(lambda x: list(x), pc2.read_points(velodyne , field_names=("x","y","z"),skip_nans=True)))
        self.lidar_input_flag = True

    def project_point_cloud(self):

        now_time = time.time() 
        try:
            img = CV_BRIDGE.imgmsg_to_cv2(self.img_msg, 'bgr8')
            # img = np.frombuffer(self.img_msg.data, dtype=np.uint8).reshape(self.img_msg.height, self.img_msg.width, -1)
        except CvBridgeError as e: 
            rospy.logerr(e)
            return
        
        # points3D =  np.asarray(self.obs_xyz)
        points3D = np.array(self.obs_xyz)

        # Filter points in front of camera
        inrange = np.where((points3D[:, 2] > -2) &
                        (points3D[:, 2] < 6) &
                        (points3D[:, 0] < 4) &
                        (points3D[:, 0] > 0.1) &
                        (np.abs(points3D[:, 1]) < 3))

        points3D = points3D[inrange[0]]

        print("len",len(points3D))

        # #voxel화 & roi설정    #########
        # points3D=list(map(lidar_module.voxel_roi_map,points3D))
        points3D=list(map(lidar_module.new_voxel_roi_map,points3D))
        points3D=list(set(points3D))
                      
        # #ransac 돌리기
        points3D=lidar_module.ransac(points3D)

        #z값 압축
        points3D=lidar_module.z_compressor(points3D[0])

        #DBSCAN 돌리기
        points3D,labels = lidar_module.dbscan(points3D)
        
        points3D = np.array(points3D)

        # 군집별로 평균 계산
        clusters = np.unique(labels)
        cluster_means = []
        for cluster in clusters:
            cluster_points = points3D[labels == cluster]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_means.append(cluster_mean)

        print("lol:",len(cluster_means))

        cluster_means = np.array(cluster_means)

        #input : np.array
        xyc = LC.transform_lidar2cam(points3D)
        xyi, _ = LC.project_pts2ing(xyc) 

        xyc2 = LC.transform_lidar2cam(cluster_means)
        xyi2, idx = LC.project_pts2ing(xyc2) 

        print("hihi:",cluster_means[idx])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2

        for i in range(len(xyi)):
            cv2.circle(img, tuple(xyi[i]), 2, (255,255,255), -1)

        for i in range(len(xyi2)):
            cv2.circle(img, tuple(xyi2[i]), 7, (0,255,255), -1)

            vector = list(cluster_means[idx[i]])
            text = f"[{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]"
            cv2.putText(img, text, tuple(xyi2[i]), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        try:
            cv2.imshow('sss' ,img)
            cv2.waitKey(1)
        except CvBridgeError as e: 
            rospy.logerr(e)

        print("realtime:",time.time()-now_time) 

if __name__ == '__main__':
    
    rospy.init_node('lidar_camera_node')
    lc = lidar2cam()
    rospy.sleep(1)
    while not rospy.is_shutdown():
        if lc.lidar_input_flag == True:
            lc.project_point_cloud()
            lc.lidar_input_flag = False