#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import paddlehub as hub
from paddlehub.common.logger import logger
import time
import math
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import socket

# parameters for mean filter
POINTS_NUM_LANDMARK = 68
windowlen_1 = 3
queue3D_points = np.zeros((windowlen_1,POINTS_NUM_LANDMARK,2))

boxPoints3D = np.array(([500., 500., 500.],
                         [-500., 500., 500.],
                         [-500., -500., 500.],
                         [500., -500., 500.],
                         [500., 500., -500.],
                         [-500., 500., -500.],
                         [-500., -500., -500.],
                         [500., -500., -500.]))
boxPoints2D = np.zeros((1,1,8,2))

class HeadPostEstimation(object):
    """
    头部姿态识别
    """

    def __init__(self, face_detector=None):
        self.module = hub.Module(name="face_landmark_localization", face_detector_module=face_detector)

    def get_face_landmark(self, image):
        """
        预测人脸的68个关键点坐标
        images(ndarray): 单张图片的像素数据
        """
        try:
            # 选择GPU运行，use_gpu=True，并且在运行整个教程代码之前设置CUDA_VISIBLE_DEVICES环境变量
            res = self.module.keypoint_detection(images=[image], use_gpu=True)
            return True, res[0]['data'][0]
        except Exception as e:
            logger.error("Get face landmark localization failed! Exception: %s " % e)
            return False, None

# Smooth filter
def mean_filter_for_landmarks(landmarks_orig):
    for i in range(windowlen_1-1):
        queue3D_points[i,:,:] = queue3D_points[i+1,:,:]
    queue3D_points[windowlen_1-1,:,:] = landmarks_orig
    landmarks = queue3D_points.mean(axis = 0)
    return landmarks

# Get feature_parameters of facial expressions
def get_feature_parameters(landmarks):
    d00 =np.linalg.norm(landmarks[27]-landmarks[8]) # Length of face (eyebrow to chin)
    d11 =np.linalg.norm(landmarks[0]-landmarks[16]) # width of face
    d_reference = (d00+d11)/2
    # Left eye
    d1 =  np.linalg.norm(landmarks[37]-landmarks[41])
    d2 =  np.linalg.norm(landmarks[38]-landmarks[40])
    # Right eye
    d3 =  np.linalg.norm(landmarks[43]-landmarks[47])
    d4 =  np.linalg.norm(landmarks[44]-landmarks[46])
    # Mouth width
    d5 = np.linalg.norm(landmarks[51]-landmarks[57])
    # Mouth length
    d6 = np.linalg.norm(landmarks[60]-landmarks[64])
    
    leftEyeWid = ((d1+d2)/(2*d_reference) - 0.02)*6
    rightEyewid = ((d3+d4)/(2*d_reference) -0.02)*6
    mouthWid = (d5/d_reference - 0.13)*1.27+0.02
    mouthLen = d6/d_reference

    return leftEyeWid, rightEyewid, mouthWid,mouthLen

# Pose estimation: get rotation vector and translation vector           
def get_pose_estimation(img_size, image_points ):
    # 3D model points
#    model_points = np.array([
#                                (0.0, 0.0, 0.0),             # Nose tip
#                                (0.0, -330.0, -65.0),        # Chin
#                                (-225.0, 170.0, -135.0),     # Left eye left corner
#                                (225.0, 170.0, -135.0),      # Right eye right corner
#                                (-150.0, -150.0, -125.0),    # Left Mouth corner
#                                (150.0, -150.0, -125.0)      # Right mouth corner
#                             
#                            ])
    
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corner
                                (-349.0, 85.0, -300.0),      # Left head corner
                                (349.0, 85.0, -300.0)        # Right head corner
                             
                            ])
    # Camera internals     
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )     
    # print("Camera Matrix:\n {}".format(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    imagePoints = np.ascontiguousarray(image_points[:,:2]).reshape((6,1,2))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, imagePoints, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_DLS)
    
    ############################
    # rotation_vector[0] = kalman_filter_simple(rotation_vector[0], 0.1, 0.01)
    # rotation_vector[1] = kalman_filter_simple(rotation_vector[1], 0.1, 0.01)
    # rotation_vector[2] = kalman_filter_simple(rotation_vector[2], 0.1, 0.01)

    # print("Rotation Vector:\n {}".format(rotation_vector))
    # print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

# Convert rotation_vector to quaternion
def get_quaternion(rotation_vector):
        # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    # theta = mean_filter_simple(theta)
    
    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    return round(w,4), round(x,4), round(y,4), round(z,4)

def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background
    
    

    h, w = overlay.shape[0], overlay.shape[1]

    if h + y < 0 or w + x < 0:
        return background

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if x < 0:
        w = w + x
        x = 0
        overlay = overlay[:, -w:]
    
    if y < 0:
        h = h + y
        y = 0
        overlay = overlay[-h:, :]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    # overlay_image = overlay[..., :3]
    # mask = overlay[..., 3:] / 255.0

    # background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image
    mask = overlay
    index = overlay[:,:,3] != 0
    index = np.repeat(index[:,:,np.newaxis], axis=2, repeats=3)
    background[y:y+h, x:x+w,:][index] = mask[:,:,:3][index]

    return background


def crop_im(im, padding=0.01):
    """
    Takes cv2 image, im, and padding % as a float, padding,
    and returns cropped image.
    """
    bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rows, cols = bw.shape
    non_empty_columns = np.where(bw.min(axis=0)<255)[0]
    non_empty_rows = np.where(bw.min(axis=1)<255)[0]
    cropBox = (min(non_empty_rows) * (1 - padding),
                min(max(non_empty_rows) * (1 + padding), rows),
                min(non_empty_columns) * (1 - padding),
                min(max(non_empty_columns) * (1 + padding), cols))
    cropped = im[int(cropBox[0]):int(cropBox[1])+1, int(cropBox[2]):int(cropBox[3])+1 , :]

    return cropped



# Socket Connect
try:
    client = socket.socket()
    client.connect(('127.0.0.1',1755))
except:
    print("\nERROR: No socket connection.\n")
    sys.exit(0)
print("Start\n")



# 创建一个video capture的实例
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap  = cv2.VideoCapture('./test2.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 将预测结果写成视频
video_writer = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)


head_post = HeadPostEstimation()

# initialize PARAMETERS
face_landmark = np.zeros((POINTS_NUM_LANDMARK,2))
landmark_shape_last = np.zeros((POINTS_NUM_LANDMARK,2))

while(True):
    start_time = time.time()
    ret, img = cap.read()
    if not ret:
        print('read frame failed')
        break

    size = img.shape
    show_img = copy.deepcopy(img)

    # 
    success, face_landmark = head_post.get_face_landmark(img)
    if success != True:
        print('ERROR: get_image_points failed')
        # continue
        face_landmark = landmark_shape_last
    landmark_shape_last = face_landmark

    # filter
    face_landmark = mean_filter_for_landmarks(face_landmark)


    for i in range(68):
        cv2.circle(img, (int(face_landmark[i][0]), int(face_landmark[i][1])),2,(0,255,0), -1, 8)
    
    # Get mouth eye data
    leftEyeWid, rightEyewid, mouthWid,mouthLen = get_feature_parameters(face_landmark)
    parameters_str = 'leftEyeWid:{}, rightEyewid:{}, mouthWid:{}, mouthLen:{}'.format(leftEyeWid, rightEyewid, mouthWid, mouthLen)
    # print(parameters_str)

    # Get quaternion
    # image_points = head_post.get_image_points_from_landmark(face_landmark)
    image_points = np.vstack((face_landmark[30],face_landmark[8],face_landmark[36],face_landmark[45],face_landmark[1],face_landmark[15]))
    success, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size, image_points)

    if success != True:
        print('ERROR: get_pose_estimation failed')
        continue
    used_time = time.time() - start_time
    print("used_time:{} sec".format(round(used_time, 3)))

    # Convert rotation_vector to quaternion
    w,x,y,z = get_quaternion(rotation_vector)
    quaternion_str = 'w:{}, x:{}, y:{}, z:{}'.format(w, x, y, z)
    # print(quaternion_str)

    # Packing data and transmit to server through Socket
    data = str(translation_vector[0,0])+':'+str(translation_vector[1,0])+':'+str(translation_vector[2,0])+':'+str(w)+':'+str(x)+':'+str(y)+':'+str(z)+':'+str(leftEyeWid)+':'+str(rightEyewid)+':'+str(mouthWid)+':'+str(mouthLen)
    try:
        client.send(data.encode('utf-8'))
    except:
        print("\nSocket connection closed.\n")
        break

    #============================================================================
    # For visualization only (below)
    #============================================================================
    
    # Project a 3D point set onto the image plane
    # We use this to draw a bounding box
        
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    for i in range(8):
        (boxPoints2D[:,:,i,:], jacobian) = cv2.projectPoints(np.array([boxPoints3D[i]]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)        
    boxPoints2D =  boxPoints2D.astype(int)

    for p in image_points:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    boxset_1 = boxPoints2D[0,0,0:4,:]
    boxset_2 = boxPoints2D[0,0,4:8,:]
    boxset_3 = np.vstack((boxPoints2D[0,0,0,:],boxPoints2D[0,0,4,:]))
    boxset_4 = np.vstack((boxPoints2D[0,0,1,:],boxPoints2D[0,0,5,:]))
    boxset_5 = np.vstack((boxPoints2D[0,0,2,:],boxPoints2D[0,0,6,:]))
    boxset_6 = np.vstack((boxPoints2D[0,0,3,:],boxPoints2D[0,0,7,:]))
    cv2.polylines(img, [boxset_1], True, (255,0,0), 3)
    cv2.polylines(img, [boxset_2], True, (255,0,0), 3)
    cv2.polylines(img, [boxset_3], True, (255,0,0), 3)
    cv2.polylines(img, [boxset_4], True, (255,0,0), 3)
    cv2.polylines(img, [boxset_5], True, (255,0,0), 3)
    cv2.polylines(img, [boxset_6], True, (255,0,0), 3)
    
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
    cv2.line(img, p1, p2, (0,255,0), 2)
    cv2.imshow("MyWindow", img)

    # receive img from unity
    fs = client.recv(1000000)
    # print(len(fs))
    _img = cv2.imdecode(np.frombuffer(fs, np.uint8), cv2.IMREAD_UNCHANGED)
    # cv2.imwrite("test.png", _img)

    # get suited img size
    mask = crop_im(_img)
    h, w = mask.shape[0], mask.shape[1]
    # new_h = 1.5*2*(image_points[1][1] - image_points[0][1])
    # new_w = w*(new_h/h)
    new_w = 3.0*(image_points[5][0] - image_points[4][0])
    new_h = h*(new_w/w)
    _img = cv2.resize(mask,(int(new_w),int(new_h)),interpolation=cv2.INTER_CUBIC) 

    dx = image_points[0][0] - 0.5*_img.shape[1]
    dy = image_points[0][1] - 0.5*_img.shape[0]

    show_img = overlay_transparent(show_img, _img, int(dx), int(dy)-30)

    cv2.imshow("Output", show_img)
    video_writer.write(show_img)

    # 按q键即可退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    used_time = time.time() - start_time
    # fps = 1.0/used_time
    print("used_time:{} sec".format(round(used_time, 3)))


client.close() # Socket disconnect
cap.release()
video_writer.release()
cv2.destroyAllWindows()