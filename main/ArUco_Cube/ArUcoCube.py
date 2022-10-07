import numpy as np
import cv2 as cv
from cv2 import aruco

#These commands know where to find the aruco numbers for that particular type of aruco marker
marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
param_markers = aruco.DetectorParameters_create()

#Loading the calibration and distortion arrays local to the camera being used here, should be changed depending on camera 
calibration_matrix =np.array([[6.53938242e+03, 0.00000000e+00, 3.37089168e+02],
                    [0.00000000e+00, 6.35503985e+03, 2.91097234e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_matrix = np.array([[7.13246703e+01, -4.12652184e+04,  3.03946545e-01, -7.06275649e-02, -1.42744624e+02]])

cap = cv.VideoCapture(0)

while True :
    
    ret, frame = cap.read()
    
    #Binarizing the image to better detect the markers
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(grey, marker_dict, parameters=param_markers)

    if marker_IDs is not None:
	
	    #Drawing the detected markers
        frame = aruco.drawDetectedMarkers(frame, marker_corners, marker_IDs)
        
        #Calculating the list of rotation(rvec) and translational(tvec) vectors to place on the marker
        rvec_list, tvec_list, objectpoints = aruco.estimatePoseSingleMarkers(marker_corners, 100, calibration_matrix, dist_matrix)
        rvec = rvec_list[0][0]
        tvec = tvec_list[0][0]
        cv.drawFrameAxes(frame, calibration_matrix, dist_matrix, rvec, tvec, 100)
	
	#Using translational vector to get the distance in 3D from the camera (coordinate system)
    tvec_str = "x=%4.0f y=%4.0f z=%4.0f"%(tvec[0],tvec[1],tvec[2])
    cv.putText(frame,tvec_str, (20,460), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2, cv.LINE_AA)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q') :
        break
    
cap.release()
cv.destroyAllWindows()
