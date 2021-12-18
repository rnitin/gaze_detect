# Import packages and libraries

import numpy as np
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import dlib
from scipy.io import loadmat

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from utils_webcam import find_landmarks, get_normalized, estimate_gaze, show_proj_gaze, show_landmarks

### CONFIG ###
device = "cpu" # change to "cuda" for CUDA inference using GPU

cap = cv2.VideoCapture(0) # capture frames from webcam 
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); # set camera width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480); # set camera height

### END CONFIG ###


### INITIALIZATION ###
dlib_detector = dlib.get_frontal_face_detector() # initialize Dlib face detector
dlib_predictor = dlib.shape_predictor("../assets/shape_predictor_68_face_landmarks.dat") # load landmark detector model

facemodel = loadmat("../assets/6 points-based face model.mat")["model"] # load MPIIGAZE 6 point 3D face model
camera_mat = np.load("../assets/camera_mat.npy") # load camera matrix
#dist_coef = np.load("../assets/dist_coef.npy") # load distortion coef -- affects performance
dist_coef = np.zeros((1, 5))
### END INITIALIZATOIN


### TORCH ###
class GazeNet(nn.Module): # define neural network model class
    # input: 30x60x1 image
    def __init__(self):
        super(GazeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5) # op = 32x56x20
        self.conv2 = nn.Conv2d(20, 50, 5) # op = 12x24x50
        self.fc1 = nn.Linear(6*12*50, 128)  # op = 500
        self.reg = nn.Linear(128,2) # op = 2
        self.float()
    def forward(self, x, y):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # op = 16x28x20
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # op = 6x12x50
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.reg(x) # append head pose
        return x
gazenet = GazeNet()
gazenet = torch.load("../assets/models/gazenet_g_lenet.pt", map_location=device) # load pretrained model
tfm = transforms.Compose([transforms.ToTensor()])
### END TORCH ###

def convert_polar_vector_np(angles):
    y = -1 * math.sin(angles[0]) # first column of angles is pitches
    x = -1 * math.cos(angles[0]) * math.sin(angles[1])
    z = -1 * math.cos(angles[0]) * math.cos(angles[1])

    mag_v = math.sqrt(x*x + y*y + z*z)
    x /= mag_v
    y /= mag_v
    z /= mag_v
    return np.array([x, y, z])

def convert_polar_vector(angles):
    y = -1 * torch.sin(angles[:,0]) # first column of angles is pitches
    x = -1 * torch.cos(angles[:,0]) * torch.sin(angles[:,1])
    z = -1 * torch.cos(angles[:,0]) * torch.cos(angles[:,1])

    mag_v = torch.sqrt(x*x + y*y + z*z)
    x /= mag_v
    y /= mag_v
    z /= mag_v
    return x, y, z


### END FUNCTION DEF ###

while(True):
    retval, img = cap.read() # read image from webcam
    img_h, img_w = img.shape[:2] # find dimension of image
    landmarks = find_landmarks(img, dlib_detector, dlib_predictor) # detect landmarks in face with Dlib
    if landmarks.shape[0] == 0: # no facial features found
        continue # skip iteration

    else: # process facial features
        #show_landmarks(landmarks, img)
        norm_img_l, hr_vec_norm_l, ht_vec_l = get_normalized(img, facemodel, landmarks, camera_mat, dist_coef, "l") # get normalized left eye image
        norm_img_r, hr_vec_norm_r, ht_vec_r = get_normalized(img, facemodel, landmarks, camera_mat, dist_coef, "r") # get normalized right eye image
        #cv2.imshow("left eye", norm_img_l)
        #cv2.imshow("right eye", norm_img_r)

        g_vec_l = estimate_gaze(norm_img_l, hr_vec_norm_l, gazenet, device) # compute left eye gaze
        g_vec_r = estimate_gaze(norm_img_r, hr_vec_norm_r, gazenet, device) # compute right eye gaze
        show_proj_gaze(g_vec_l, g_vec_r, landmarks, img) # display 2D gaze projection on webcam image
        
        if cv2.waitKey(10) == ord("q"): # quit on pressing q
            break

cap.release()
cv2.destroyAllWindows()
