# Import packages and libraries

import numpy as np
import math
import cv2
from PIL import Image
import torch
from torchvision import transforms


from utils_common import convert_polar_vector, convert_polar_vector_np

### TORCH ###
tfm = transforms.Compose([transforms.ToTensor()])
### END TORCH ###

### FUNCTION DEF ###

##### 
## Find landmark locations in image using DLib
#####
def find_landmarks(cv2_img, dlib_detector, dlib_predictor):
    img_dlib = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    det = dlib_detector(img_dlib, 1) # upsample image once and detect bounds of face
    img_disp_dlib = cv2_img.copy() # copy image for display
    landmarks = []
    for k, d in enumerate(det): # find coordinates of six landmarks
        img_disp_dlib = cv2_img.copy()
        shape = dlib_predictor(img_dlib, det[0]) # find features assuming only one face will be present
        for i in [36, 39, 42, 45, 48, 54]: # order: rer, rel, ler, lel, mr, me
            landmarks.append([shape.part(i).x, shape.part(i).y]) # append relative coords of features
    landmarks = np.array(landmarks)
    return landmarks


##### 
##  Normalize eye image from webcam image given eye center
#####
def normalize_image(img, e_c, hr_mat, e_img_w, e_img_h, camera_mat):
    focal_new = 960 # parameter from paper authors
    dist_new = 600 # parameter from paper authors
    
    dist = np.linalg.norm(e_c)
    z_scale =  dist_new / dist
    cam_new = np.array([[focal_new, 0 ,e_img_w/2], [0, focal_new, e_img_h/2], [0, 0, 1]])
    scale_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]])
    forward = e_c / dist
    hr_mat_x = hr_mat[:,0]
    down = np.cross(forward, hr_mat_x)
    down = down / np.linalg.norm(down)
    right =  np.cross(down, forward)
    right = right / np.linalg.norm(right)
    rot_mat = np.array([right, down, forward])
    warp_mat = (cam_new @ scale_mat) @ (rot_mat @ np.linalg.inv(camera_mat))
    
    img_warp = cv2.warpPerspective(img, warp_mat, (e_img_w, e_img_h))
    img_warp = cv2.cvtColor(img_warp, cv2.COLOR_RGB2GRAY)
    img_warp = cv2.equalizeHist(img_warp)

    cnv_mat = scale_mat * rot_mat
    hr_mat_new = cnv_mat *hr_mat
    hr_vec_norm, _ = cv2.Rodrigues(hr_mat_new)
    #ht_vec_norm = cnv_mat * e_c
    return img_warp, hr_vec_norm

##### 
## Obtain normalized eye image and head pose
#####
def get_normalized(cv2_img, facemodel, landmarks, camera_mat, dist_coef, eye_choice):
    _, hr_vec, ht_vec = cv2.solvePnP(facemodel.T, landmarks.astype(float), camera_mat, dist_coef, flags=0) # get 3D head pose
    hr_mat, _ = cv2.Rodrigues(hr_vec) # head rotation matrix
    f_c = hr_mat @ facemodel
    f_c = f_c + np.tile(ht_vec.reshape(3,1), (1,6)) # transformed feature coordinates
    if eye_choice == "l":
        e_c = (f_c[:,2]+f_c[:,3])/2 # left eye center
    else:
        e_c = (f_c[:,0]+f_c[:,1])/2 # right eye center
    e_img_w = 60 # desired eye image width = 60
    e_img_h = 36 # desired eye image width = 36
    img_norm, hr_vec_norm = normalize_image(cv2_img, e_c, hr_mat, e_img_w, e_img_h, camera_mat) # normalized eye img and head rot vec
    img_norm = cv2.flip(img_norm, 0) # flip horizontally

    hr_mat_norm, _ = cv2.Rodrigues(hr_vec_norm) # get norm rotation matrix
    z_vec_norm = hr_mat_norm[:,2]
    h_theta = math.asin(z_vec_norm[1]) # vertical head pose angle
    h_phi = math.atan2(z_vec_norm[0], z_vec_norm[2]) # horizontal head pose angle
    h_pose_norm = convert_polar_vector_np([h_theta, h_phi]) # unit head pose vector

    return img_norm, h_pose_norm, ht_vec

##### 
## Estimate gaze vector in eye frame from normalized eye image
#####
def estimate_gaze(img_norm, h_pose_norm, gazenet, device="cpu"):
    pil_img = Image.fromarray(img_norm) # convert to PIL image
    ip_tensor = tfm(pil_img).unsqueeze(dim = 0) # input eye image tensor
    h_tensor = torch.tensor(h_pose_norm).unsqueeze(dim=0) # input head pose tensor
    if device != "cpu":
        ip_tensor = ip_tensor.to(device)
        h_tensor = h_tensor.to(device)
        out = gazenet(ip_tensor, h_tensor).cpu() # get eye gaze angles
        pred = torch.stack(convert_polar_vector(out), dim = 1) # get eye gaze unit tensor
    else:
        out = gazenet(ip_tensor, h_tensor) # get eye gaze angles
        pred = torch.stack(convert_polar_vector(out), dim = 1) # get eye gaze unit tensor
    g_vec = pred.squeeze().detach().numpy()  # eye gaze unit vector
    g_vec[0] *= -1 # img flipped horizontally before training
    return g_vec

##### 
## Project and display gaze vector in eye frame
#####
def show_proj_gaze(g_vec_l, g_vec_r, landmarks, img):
    img_show = img.copy()
    r_e_c = ((landmarks[0] + landmarks[1])/2).astype(int)
    l_e_c = ((landmarks[2] + landmarks[3])/2).astype(int)
    g_vec_l = g_vec_l*100
    g_vec_l[0] *= -1 # img flipped horizontally
    g_vec_l = g_vec_l.astype(int)
    g_vec_r = g_vec_r*100 # arbitrary scaling of unit length vector for visualization
    g_vec_r[0] *= -1 # img flipped horizontally
    g_vec_r = g_vec_r.astype(int)
    for i in range(6): # display landmarks
        img_show = cv2.circle(img_show, (landmarks[i][0], landmarks[i][1]), radius=0, color=(0, 0, 255), thickness=2)

    img_show = cv2.line(img_show, (l_e_c[0], l_e_c[1]), (l_e_c[0] + g_vec_l[0], l_e_c[1] + g_vec_l[1]), color=(0, 255, 0), thickness=1)
    img_show = cv2.line(img_show, (r_e_c[0], r_e_c[1]), (r_e_c[0] + g_vec_r[0], r_e_c[1] + g_vec_r[1]), color=(255, 100, 100), thickness=1)
    img_show = cv2.flip(img_show, 1)
    cv2.imshow("gaze projection", img_show) 

##### 
## Display obtained landmarks from Dlib
#####
def show_landmarks(landmarks, img):
    img_show = img.copy()
    for i in range(6): # display landmarks
        img_show = cv2.circle(img_show, (landmarks[i][0], landmarks[i][1]), radius=0, color=(0, 0, 255), thickness=2)
    img_show = cv2.flip(img_show, 1)
    cv2.imshow("face landmarks dlib", img_show)

### END FUNCTION DEF ###
