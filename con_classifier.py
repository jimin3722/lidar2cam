import numpy as np
import torch
import cv2
import time
import os, sys
from cv_bridge import CvBridge, CvBridgeError
from lidar2cam_fish import LIDAR2CAMTransform
import rospy

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

class Cone_Classfier():

    def __init__(self):
        self.WEIGHT_PATH="/home/jimin/catkin_ws/src/lidar_camera_calibration/lidar2cam/best_ep_100.pt" # msi
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path = self.WEIGHT_PATH, force_reload=True)

    def yolo_detecter(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model([img], 416)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # [ xmin | ymin | xmax | ymax | conf | cls ]
        img_cones = [[int(i[0]),int(i[1]),int(i[2]),int(i[3]),int(i[4]),int(i[5])] for i in results.xyxy[0] if i[4] > 0.5 ]
        print("imgc",img_cones)
        print("len",len(img_cones))
        bbox = results.xyxy[0]
        if len(bbox) > 0:
            print("----------------------------------------")
            print('obj_num:{}'.format(len(bbox)))
            for i in bbox:
                if i[4] > 0.5:
                    print('xmin:{}, ymin:{}, xmax:{}, ymax:{}, cls:{}, conf={}'.format( int(i[0]), int(i[2]), int(i[1]), int(i[3]), results.names[int(i[5])], int(i[4]*100) ))
                    cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255,0,0), 3 )
                    cv2.putText(img,results.names[int(i[5])],(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                print("----------------------------------------")
        else:
            pass
        try:
            cv2.imshow('sss1' ,img)
            cv2.waitKey(1)
        except CvBridgeError as e: 
            rospy.logerr(e) 
        return img_cones
    
     
    def cone_classifier(self, cones, img):
      
        cones = np.array(cones)
        classified_cones = np.zeros(cones.shape[0], dtype=int)
        xyc2 = LC.transform_lidar2cam(cones)
        projected_cones, cone_idx = LC.project_pts2ing(xyc2) 
        img = cv2.undistort(img, camera_matrix, dist_coeffs, None)
        t=time.time()
        img_cones = self.yolo_detecter(img)
        print("real_time",time.time()-t)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t=time.time()
        results = self.model([img], size=416)
        print("real_time",time.time()-t)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # [ xmin | ymin | xmax | ymax | conf | cls ]
        img_cones = [[int(i[0]),int(i[1]),int(i[2]),int(i[3]),int(i[4]),int(i[5])] for i in results.xyxy[0] if i[4] > 0.5 ]

        for img_cone in img_cones:
            closest_dist = 10000
            closest_idx = 9999
            img_cone_vector = [ int((img_cone[0]+img_cone[2])/2), int((img_cone[1]+img_cone[3])/2) ]
            
            for idx, projected_cone in enumerate(projected_cones):
                distance = np.linalg.norm(projected_cone - img_cone_vector)
                
                if closest_dist > distance :
                    closest_dist = distance
                    closest_idx = idx
                    closest_cone = projected_cone
            print("dist",closest_dist)
            
            if closest_idx != 9999 and classified_cones[cone_idx[closest_idx]] == 0 and img_cone[0] < closest_cone[0] < img_cone[2] and img_cone[1] < closest_cone[1] < img_cone[3]:
                if img_cone[5] == 0:
                    classified_cones[cone_idx[closest_idx]] = 1
                if img_cone[5] == 1:
                    classified_cones[cone_idx[closest_idx]] = 2

        print("cc",classified_cones)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2

        for i in range(len(projected_cones)):

            if classified_cones[i] == 2: 
                cv2.circle(img, tuple(projected_cones[i]), 7, (0,255,255), -1)
            elif classified_cones[i] == 1: 
                cv2.circle(img, tuple(projected_cones[i]), 7, (255,0,0), -1)
            else: 
                cv2.circle(img, tuple(projected_cones[i]), 7, (255,255,255), -1)
            vector = list(cones[i])
            print(vector)
            text = f"[{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]"
            #cv2.putText(img, text, tuple(projected_cones[i]), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        try:
            cv2.imshow('sss' ,img)
            cv2.waitKey(1)
        except CvBridgeError as e: 
            rospy.logerr(e)

        return classified_cones
        




