import sensor_msgs.point_cloud2 as pc2
import numpy as np
from numpy import array,transpose,dot,cross
from math import acos,sin,cos
import random
from sklearn.cluster import DBSCAN
import multiprocessing

#location (4,0,0)
#roi 설정
floor_value=-5 #음수, 라이다 위치부터 아래 몇미터까지인지
roof_value=13 #위 몇미터까지인지
lenth=20 #몇미터 앞까지 볼건지
width=11 #라이다로 볼 데이터의 폭 (2x라면 왼쪽으로x만큼 오른쪽으로 x만큼)
min_intensity=0 #라이다 데이터의 최소 강도 (현재 사용하지는 않음)
back_lenth=0 #양수, 라이다기준 몇미터 뒤까지 볼건지

roi_value=[floor_value,roof_value,lenth,width,min_intensity,back_lenth]

#dbscan 설정 https://saint-swithins-day.tistory.com/m/81, https://bcho.tistory.com/1205
epsilon=0.2 #입실론 값 0.4
min_points=4 #한 군집을 만들 최소 포인트 개수 4

#voxel 설정
decimal=2 #소수점 n자리까지 데이터들을 반올림함, 따라서 데이터 간의 차이가 소수점 n+1자리수 보다 작다면 한 점으로 취급 
delta=0.01 #데이터가 delta의 배수로 나타나짐  

#ransac 설정 https://gnaseel.tistory.com/33
p_dist=0.1 #추출된 모델로부터 거리가 n이하이면 inlier로 취급함
reps=100 #ransac 반복횟수
p_angle=3.14/8 #라디안, 추출된 모델과 xy평면의 최대 각도 차이

def intensity_voxel_roi_map(raw_data):
    if -0.5*roi_value[3]<raw_data[1]<0.5*roi_value[3] and -1*roi_value[5]<raw_data[0]<roi_value[2] and roi_value[0]<raw_data[2]<roi_value[1]:
        x = (raw_data[0]//delta)*delta
        y = (raw_data[1]//delta)*delta
        z = (raw_data[2]//delta)*delta
        intensity = int(raw_data[3])
        return tuple([x,y,z,intensity])
    else:
        return tuple([0,0,0,0])    

def new_voxel_roi_map(raw_data):
    if -0.5*roi_value[3]<raw_data[1]<0.5*roi_value[3] and -1*roi_value[5]<raw_data[0]<roi_value[2] and roi_value[0]<raw_data[2]<roi_value[1]:
        x = (raw_data[0]//delta)*delta
        y = (raw_data[1]//delta)*delta
        z = (raw_data[2]//delta)*delta
        return tuple([x,y,z])
    else:
        return tuple([0,0,0])

def voxel_roi_map(raw_data):
    if -0.5*roi_value[3]<raw_data[1]<0.5*roi_value[3] and -1*roi_value[5]<raw_data[0]<roi_value[2] and roi_value[0]<raw_data[2]<roi_value[1]:
        x = round(raw_data[0],decimal)
        y = round(raw_data[1],decimal)
        z = round(raw_data[2],decimal)
        return tuple([x,y,z])
    else:
        return tuple([0,0,0])
    

def ransac(input_data):
    input_array=array(input_data)
    best_count=100000
    for j in range(reps):
        random_elements=random.sample(input_data,3)
        #뽑은 점 3개로 평면 만들기 위해 np array로 변경
        p1= array(random_elements[0])
        p2= array(random_elements[1])
        p3= array(random_elements[2])
        #평면을 이룰 벡터 2개 만들기
        v1=p2-p1
        v2=p3-p1
        #벡터 2개 크기 구하기
        v1_size=(v1[0]**2+v1[1]**2+v1[2]**2)**(1/2)            
        v2_size=(v2[0]**2+v2[1]**2+v2[2]**2)**(1/2)
        #벡터 2개 사잇각의 코사인 값
        cos_v1v2=dot(v1,v2)/(v1_size*v2_size)
        #이상하게 코사인인데 -1,1밖의 값이 가끔 나와서 변경
        if(cos_v1v2>1 or cos_v1v2<-1):
            cos_v1v2=0.99999
        #벡터 2개 사잇각
        angle_v1v2=acos(cos_v1v2)
        #평면의 단위법선벡터
        pppp=v1_size*v2_size*sin(angle_v1v2)
        if pppp == 0:
            pppp = 0.00001
        normal_vector=cross(v1,v2)/pppp
        #이것도 가끔 단위벡터의 z값인데 -1,1밖의 값이 나와서 변경
        if(normal_vector[2]>1 or normal_vector[2]<-1):
            normal_vector[2]=0.99999  
        
        #샘플링해서 구한 평면과 xy평면 사잇각의 크기      
        angle=acos(normal_vector[2])
        if angle<p_angle or angle>3.14-p_angle:
            ransac_count=0                
            #data 벡터화
            vector_array=input_array-p1
            #평면과의 거리 계산
            dist_array=abs(dot(vector_array,normal_vector))
            #평면 제거된 모델 추출
            weight_list=np.where(dist_array>p_dist,True,False)
            ransac_count=np.sum(weight_list)

            #센 점 개수로 가장 좋은 모델 추출
            if ransac_count < best_count:
                best_count=ransac_count
                best_weight_list=weight_list 

    best_no_land_model=input_array[best_weight_list]
    return best_no_land_model, best_weight_list


def z_compressor(input_data):
    def z_com(input_point):
        input_point[2]=input_point[2]*epsilon*10000/(lenth*456)
        return input_point
    input_data=list(map(z_com,input_data))
    return input_data


def dbscan(input_data):
    #eps과 min_points가 입력된 모델 생성
    model=DBSCAN(eps=epsilon, min_samples=min_points)
    #데이터를 라이브러리가 읽을 수 있게 np array로 변환
    DB_Data=array(input_data)
    #모델 예측
    labels=model.fit_predict(DB_Data)
    k=0
    no_noise_model=[]
    no_noise_label=[]
    for i in input_data:
        if labels[k] != -1 :
            z=i[2]*(lenth*456)/(epsilon*10000)
            no_noise_model.append([i[0],i[1],z])
            no_noise_label.append(labels[k])
        k+=1
    return no_noise_model, no_noise_label

def intensity_dbscan(input_data,intensity_list):
    #eps과 min_points가 입력된 모델 생성
    model=DBSCAN(eps=epsilon, min_samples=min_points)
    #데이터를 라이브러리가 읽을 수 있게 np array로 변환
    DB_Data=array(input_data)
    #모델 예측
    labels=model.fit_predict(DB_Data)
    k=0
    no_noise_model=[]
    no_noise_label=[]
    no_noise_intensity=[]
    for i in input_data:
        if labels[k] != -1 :
            z=i[2]*(lenth*456)/(epsilon*10000)
            no_noise_model.append([i[0],i[1],z])
            no_noise_label.append(labels[k])
            no_noise_intensity.append(intensity_list[k])
        k+=1
    return no_noise_model, no_noise_label, no_noise_intensity

def tf2tm(no_z_points,x,y,heading):
    obs_tm=np.empty((1,3))
    T = [[cos(heading), -1*sin(heading), x], \
            [sin(heading),  cos(heading), y], \
            [      0     ,      0       , 1]] 
    for point in no_z_points:
        obs_tm = np.append(obs_tm,[dot(T,transpose([point.x+4, point.y,1]))],axis=0) # point[0] -> 객체를 subscribe할 수 없음 오류
    obs_tm[:,2]=0
    return obs_tm