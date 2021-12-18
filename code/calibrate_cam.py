# Import packages and libraries

import cv2
import numpy as np
import glob

cap = cv2.VideoCapture(0) # capture frames from webcam 
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); # set camera width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480); # set camera height
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # type = iter and eps both, max iter = 30, min acc = 0.001

obj_pt = np.zeros((9*6, 3), np.float32) # define format for obj pts
obj_pt[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1, 2) # arrange XY coords of obj pts in order of findchessboardcorners

obj_pts = [] # list to store all obj pts
img_pts = [] # list to store all computed img pts

count = 0

while(count < 100): # capture 100 chessboard frames
	retval, img = cap.read() # capture img from webcam
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert frame to grayscale
	retval, corners = cv2.findChessboardCorners(gray, (9,6),None)
	if retval == True:
		corners_new = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) # search = 5x5 around corner, reject = none, term = criteria
		obj_pts.append(obj_pt)
		img_pts.append(corners_new)
		cv2.drawChessboardCorners(img, (9, 6), corners_new, retval)
		cv2.imshow("chessboard corners", img)
		cv2.waitKey(50)
	count += 1
# End after 100 images are captured
cap.release()
cv2.destroyAllWindows()

retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None) # calibrate camera
print("Camera matrix: ")
print(mtx)
print("Distortion coff: ")
print(dist)

np.save("../assets/camera_mat.npy", mtx) # save camera matrix
np.save("../assets/dist_coef.npy", mtx) # save distortion coeff