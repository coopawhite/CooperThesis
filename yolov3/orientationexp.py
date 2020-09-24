import numpy as np
import statistics
import cv2
import argparse
import time
import os
import subprocess
import pyrealsense2 as rs
from math import atan2, cos, sin, sqrt, pi

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    ## [visualization1]

def getOrientation(pts, img):

    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    ## [visualization]

    return angle+pi/2

    #set up stream


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

try:
    while True:

        
        frames = pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())


        contour = np.copy(color_image)

        #---------- HSV MASKING -----------#

        # Convert image to hsv
        hsv = cv2.cvtColor(contour, cv2.COLOR_BGR2HSV) 
        #lets get some values for our hsv thresholds, what colours should we play between?

        weaker = np.array([0, 115, 65]) #kind of a pale yellow colour
        stronger = np.array([59, 255, 255]) #strong, vibrant yellow.

        #make a mask

        hsvmask = cv2.inRange(hsv, weaker, stronger) 

        #----------- CROP MASKING -------------#

        recmask = np.zeros_like(hsvmask)
        cv2.rectangle(recmask, (640 - 250, 450 - 250), (640 + 250, 450 + 250), (255, 255, 255), -1)
        mask = cv2.bitwise_and(hsvmask, recmask)
        
        #mask_vis = cv2.hconcat((recmask, hsvmask))

        #cv2.namedWindow('Focused Object', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow("Focused Object", mask_vis)
        

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]


        for i, c in enumerate(contours):
            # Calculate the area of each contour
            area = cv2.contourArea(c)
            # Ignore contours that are too small or too large
            if area < 1e3 or 1e5 < area:
                continue

            #   Draw each contour only for visualisation purposes
            cv2.drawContours(contour, contours, i, (0, 0, 255), 2)
            # Find the orientation of each shape
            angle = getOrientation(c, contour)
            print("Orientation is", angle*(180/pi))

        cv2.namedWindow('Contour', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Contour', contour)

        cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Mask', hsvmask)
        cv2.waitKey(1)

except KeyboardInterrupt:
    # Stop streaming

    print("\n[INFO]: Closing...")
    pipeline.stop()
    cv2.destroyAllWindows()