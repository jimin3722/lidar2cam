import numpy as np
import math

def rotationMtx(roll, pitch, yaw):
    # Roll, pitch, yaw 값을 라디안으로 변환
    # 회전 행렬 생성
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(roll), -np.sin(roll), 0],
                   [0, np.sin(roll), np.cos(roll), 0],
                   [0, 0, 0, 1]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch), 0],
                   [0, 1, 0, 0],
                   [-np.sin(pitch), 0, np.cos(pitch), 0],
                   [0, 0, 0, 1]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                   [np.sin(yaw), np.cos(yaw), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    # 회전 행렬들을 순서대로 곱함
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    return R

def translationMtx(x,y,z):
    M = np.array([[1,0,0,x],
                  [0,1,0,y],
                  [0,0,1,z],
                  [0,0,1,1]])
    return M

def transformMTX_lidar2cam():
    x_rel = 0.15
    y_rel = 0.00
    z_rel = 0.10
    # x_rel = 0
    # y_rel = 0
    # z_rel = 0
    R_T = np.matmul(translationMtx(x_rel, y_rel, z_rel),rotationMtx(np.deg2rad(-90.), 0,  0))
    R_T = np.matmul(R_T, rotationMtx(0, np.deg2rad(90), 0))
    R_T = np.matmul(R_T, rotationMtx(np.deg2rad(4.), 0, 0))
    R_T = np.matmul(R_T, rotationMtx(0, np.deg2rad(-1), 0))
    # R_T = np.matmul(R_T, rotationMtx(0, 0, np.deg2rad(0)))

    RT = np.linalg.inv(R_T)
    return RT

def project2img_mtx(h,w,f):
    fc_x = h/(2*np.tan(np.deg2rad(f/2)))
    fc_y = fc_x
    #the center of image
    cx = int(w/2)
    cy = int(h/2)
    #transformation matrix from 3D to 2D
    R_f = np.array([[fc_x,  0,     cx],
                    [0,     fc_y,  cy]])
    return R_f

class LIDAR2CAMTransform:
    def __init__(self, w, h, f):
        self.width = w
        self.height = h
        self.RT = transformMTX_lidar2cam()
        self.proj_mtx = project2img_mtx(h,w,f)
    
    def transform_lidar2cam(self, xyz_p) :
        xyz_c = np.matmul(np.concatenate([xyz_p, np.ones((xyz_p.shape[0], 1))], axis=1), self.RT.T)
        return xyz_c
    
    def project_pts2ing(self, xyz_c, crop=True):
        xyz_c = xyz_c.T
        xc, yc, zc = xyz_c[0,:].reshape([1,-1]), xyz_c[1,:].reshape([1,-1]), xyz_c[2,:].reshape([1,-1])
        xn, yn = xc/(zc+0.0001), yc/(zc+0.0001)
        xyi = np.matmul(self.proj_mtx, np.concatenate([xn, yn, np.ones_like(xn)], axis=0))
        xyi = xyi[0:2,:].T
        if crop:
            xyi, idx = self.crop_pts(xyi)
        else:
            pass
            #xyi = self.crop_pts(xyi)
        xyi = xyi.astype('int32')
        xyi = xyi.astype('int32')
        return xyi, idx

    def crop_pts(self, xyi):
        
        idx = np.where((xyi[:, 0]>=0)& 
                     (xyi[:, 0]<self.width)&
                     (xyi[:, 1]>=0)&
                     (xyi[:, 1]<self.height))

        xyi = xyi[idx[0]]
        # xyi = xyi[np.logical_and(xyi[:, 0]>=0, xyi[:, 0]<self.width), :]
        # xyi = xyi[np.logical_and(xyi[:, 1]>=0, xyi[:, 1]<self.height), :]

        return xyi, idx[0]

import math

def rpy_to_xyz(roll, pitch, yaw):
    # Degree to radian 변환
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    # X, Y, Z 각도 계산
    x = roll_rad
    y = pitch_rad
    z = yaw_rad

    # Radian to degree 변환
    x_deg = math.degrees(x)
    y_deg = math.degrees(y)
    z_deg = math.degrees(z)

    return x_deg, y_deg, z_deg

# Roll, Pitch, Yaw 각도 값 입력
roll_angle = 30
pitch_angle = 45
yaw_angle = 60

# X, Y, Z 각도 계산
x_angle, y_angle, z_angle = rpy_to_xyz(roll_angle, pitch_angle, yaw_angle)

# 결과 출력
print("X 각도:", x_angle)
print("Y 각도:", y_angle)
print("Z 각도:", z_angle)

if __name__ == "__main__":

    roll_angle = 100
    pitch_angle = -84.88
    yaw_angle = -15.70

    # X, Y, Z 각도 계산
    x_angle, y_angle, z_angle = rpy_to_xyz(roll_angle, pitch_angle, yaw_angle)

    # 결과 출력
    print("X 각도:", x_angle)
    print("Y 각도:", y_angle)
    print("Z 각도:", z_angle)
        

