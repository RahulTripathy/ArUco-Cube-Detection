import cv2 as cv
import numpy as np
import glob

#Providing the number of internal corners of chessboard (those surrounded by squares on all four sides) as well as the size of each box in mm, both unique to my image and must be changed depending on the chessboard image used for calibration
cb_length=7
cb_breadth=5
cb_size=20.5

#Ending criteria
criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

#Make object and image points
cb_3d_array = np.zeros((cb_length*cb_breadth,3),np.float32)
cb_3d_array[:,:2] = np.mgrid[0:cb_length,0:cb_breadth].T.reshape(-1,2)*cb_size
list_cb_3d_array = []
list_cb_2d_array = []

list_images = glob.glob('*.jpg')

for Cal_Image in list_images:
    frame = cv.imread(Cal_Image)

    grey = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(grey, (cb_breadth,cb_length), None)

    if ret == True:

	#Adding true values to the list of object points
        list_cb_3d_array.append(cb_3d_array)
	
	#Collecting the corners of the grid and adding it to the list of image points
        corners2 = cv.cornerSubPix(grey, corners, (11,11),(-1,-1),criteria)
        list_cb_2d_array.append(corners2)

        #Draw detected lines on chessboard
        cv.drawChessboardCorners(frame,(cb_length,cb_breadth),corners2, ret)
        cv.imshow("Drawn lines and corners",frame)
        cv.waitKey(500)

cv.destroyAllWindows

#Calculating the distortion and intrinsic and extrinsic values of the camera from the image data acquired till now
ret, mtx, dist, rvecs, tvecs, = cv.calibrateCamera(list_cb_3d_array,list_cb_2d_array,grey.shape[::-1],None,None)

print("Calibration Matrix : ")
print(mtx)
print(" Distortion : ")
print(dist)
