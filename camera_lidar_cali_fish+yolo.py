#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in modules
import torch
import os
import sys
import time
from collections import defaultdict

# External modules  
import cv2
import numpy as np

from lidar2cam_fish import LIDAR2CAMTransform
import lidar_module

import rospy
from cv_bridge import CvBridge, CvBridgeError
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2

# Global variables
TF_BUFFER = None
CV_BRIDGE = CvBridge()

# LC = LIDAR2CAMTransform(1280, 720, 44.5)
LC = LIDAR2CAMTransform(640, 480, 78)

fx = 345.727618
fy = 346.002121
cx = 320.000000
cy = 240.000000
w = -0.24000
k1 = -0.320124
k2 = 0.098598
k3 = 0
p1 = 0.001998
p2 = 0.000177

# 카메라 매트릭스
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])

# 왜곡 계수
dist_coeffs = np.array([k1, k2, k3, p1, p2])


class lidar2cam():
    def __init__(self):
        self.img_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback, queue_size = 1)
        self.lidar_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.lidar_callback, queue_size = 1)
        self.WEIGHT_PATH="/home/jimin/catkin_ws/src/lidar_camera_calibration/lidar2cam/best_ep_100.pt" # msi
        # self.WEIGHT_PATH = (os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))+"/pt/morai_last.pt"
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path = self.WEIGHT_PATH, force_reload=True)
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
        
        h, w = img.shape[:2]

        # 왜곡 보정
        img = cv2.undistort(img, camera_matrix, dist_coeffs, None)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # img = img[..., ::-1] # BGR에서 RGB로 변경
        
        results = self.model([img], size=640)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # img = img[..., ::-1] # BGR에서 RGB로 변경

        bbox = results.xyxy[0]   

        img_cones = [[int(i[0]),int(i[1]),int(i[2]),int(i[3]),int(i[4]),int(i[5])] for i in bbox if i[4] > 0.5 ]
        
        if len(bbox) > 0:
            print("----------------------------------------")
            print('obj_num:{}'.format(len(bbox)))
            for i in bbox:
                if i[4] > 0.5:
                    print('xmin:{}, ymin:{}, xmax:{}, ymax:{}, cls:{}, conf={}'.format( int(i[0]), int(i[2]), int(i[1]), int(i[3]), results.names[int(i[5])], int(i[4]*100) ))
                    cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255,0,0), 2 )
                    cv2.putText(img,results.names[int(i[5])],(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            print("----------------------------------------")
        else:
            pass

        # points3D =  np.asarray(self.obs_xyz)
        points3D = np.array(self.obs_xyz)

        # Filter points in front of camera
        inrange = np.where((points3D[:, 2] > -2) &
                        (points3D[:, 2] < 6) &
                        (points3D[:, 0] < 4) &
                        (points3D[:, 0] > 0.1) &
                        (np.abs(points3D[:, 1]) < 3))

        points3D = points3D[inrange[0]]

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

        # cone_detect
        label_points = defaultdict(list)
        for l, p in zip(labels, points3D):
            label_points[l].append(p)

        cone_centers=[]
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
            
            if x_range>0.05 and x_range<0.55 and y_range>0.05 and y_range<0.55 and z_range>0.05 and z_range<1:
                x_mean=sum(x_list)/len(x_list)
                y_mean=sum(y_list)/len(y_list)
                z_mean=sum(z_list)/len(z_list)
                cone_centers.append([x_mean,y_mean,z_mean])
            elif max(x_list)<3 and x_range>0.05 and x_range<0.55 and y_range>0.05 and y_range<0.55 and z_range<x_range/4 and z_range>0.05:
                x_mean=sum(x_list)/len(x_list)
                y_mean=sum(y_list)/len(y_list)
                z_mean=sum(z_list)/len(z_list)
                cone_centers.append([x_mean,y_mean,z_mean])   

        # rubber cones coordinate
        cones = np.array(cone_centers)

        classified_cones = np.zeros(cones.shape[0], dtype=int)

        xyc2 = LC.transform_lidar2cam(cones)
        projected_cones, cone_idx = LC.project_pts2ing(xyc2) 
        print("idx",cone_idx)

        for idx, projected_cone in zip(cone_idx, projected_cones):

            closest_dist = 9999
            
            for img_cone in img_cones:

                img_cone_vector = [ int((img_cone[0]+img_cone[2])/2), int((img_cone[1]+img_cone[3])/2) ]
                cv2.circle(img, tuple((int((img_cone[0]+img_cone[2])/2), int((img_cone[1]+img_cone[3])/2))), 5, (0,0,255), -1)
                distance = np.linalg.norm(projected_cone - img_cone_vector)
               
                if closest_dist > distance:
                    closest_dist = distance
                    closest_cone = img_cone
            
            if classified_cones[idx] == 0 and closest_cone[0] < projected_cone[0] < closest_cone[2] and closest_cone[1] < projected_cone[1] < closest_cone[3]:
                if closest_cone[5] == 0:
                    classified_cones[idx] = 1#blue
                elif closest_cone[5] == 1:
                    classified_cones[idx] = 2#yellow
                else:
                    classified_cones[idx] = 0

        print("cc",classified_cones)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2

        for i in range(len(projected_cones)):

            if classified_cones[cone_idx[i]] == 1: 
                cv2.circle(img, tuple(projected_cones[i]), 7, (255,0,0), -1)
            elif classified_cones[cone_idx[i]] == 2: 
                cv2.circle(img, tuple(projected_cones[i]), 7, (20,255,255), -1)
            else: 
                cv2.circle(img, tuple(projected_cones[i]), 7, (255,255,255), -1)
            vector = list(cones[i])
            text = f"[{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]"
            #cv2.putText(img, text, tuple(projected_cones[i]), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
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