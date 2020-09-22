import numpy as np
import cv2
import argparse
import time
import os
import subprocess
import pyrealsense2 as rs
from math import atan2, cos, sin, sqrt, pi

#------------------------------------------------#
#--------------- FUNCTION DEFS ------------------#
#------------------------------------------------#

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

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


#-------------------------------------------------#    
#------------------- CONFIG ----------------------#
#-------------------------------------------------#


print("[INFO] Python script is running..")
print("[INFO] Opening C Program...")
p = subprocess.Popen(['./data_reciever'], 
                        stdin=subprocess.PIPE, stderr=subprocess.STDOUT)

cfg_path = "cfg/yolov3.cfg"
print("[INFO] CFG read from:", cfg_path)
weights_path = "weights/yolov3.weights"
print("[INFO] Weights read from:", weights_path)


print("[INFO] loading YOLO from disk...")

#read in darknet
net = cv2.dnn.readNetFromDarknet("cfg/yolov3.cfg", "weights/yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#get all the classes
labelsPath = "data/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

#get individual colours for each class
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(LABELS), 3)) #make a colour for each class

#get class names
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#-------------------------------------------------#
#------------------- DETECTION -------------------#
#-------------------------------------------------#

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


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.depth
align = rs.align(align_to)

dec_filter = rs.decimation_filter()   # Decimation - reduces depth frame density
spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise
try:
    while True:

        start = time.time() #We want to see how long each iteration takes

        

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
       
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        aligned_depth_frame = dec_filter.process(aligned_depth_frame)
        aligned_depth_frame = spat_filter.process(aligned_depth_frame)
        aligned_depth_frame = temp_filter.process(aligned_depth_frame)
        aligned_depth_frame = aligned_depth_frame.as_depth_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
       
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

       
        # Copy across all the images we use for visualisation
        detection_img = np.copy(color_image)
        focus = np.copy(color_image)
        contour = np.copy(color_image)

        #get height width and channels
        (h, w, c) = detection_img.shape 
        aspect = w/h
        scale = h/416
        crop_start = round(416 * (aspect-1) / 2)

        #make a blob
        blob = cv2.dnn.blobFromImage(detection_img, 1/255, (416, 416),
                swapRB=True, crop=False) 

        net.setInput(blob)

        #get the values of what the network detected
        outs = net.forward(layer_names)
     
        #pre allocate some variables
        class_ids = []
        confidences = []
        boxes = []
        resultant = []
        posVal = []
        dist = 0.00

        #------------------- LOOP OVER DETECTIONS -------------------#

        for out in outs:
            for detection in out:
                
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.4 and class_id == 46: #this will disregard any detections that arent a banana
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    posVal.append([centerX-640, centerY-360])
                    # Rectangle coordinates
                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)

                    resultant.append(sqrt((centerX-640)**2 + (centerY-360)**2)) 
                    #note ^ opencv pixel co-ords are form top corner, therefore by subtracting 
                    # the middle of the frame we change the reference from the corner to the middle.

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                       

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.4) #complete non maxima supression
        font = cv2.FONT_HERSHEY_PLAIN

        #draw the boxes
        for i in range(len(boxes)):
            if (i in indexes):
                x, y, w, h = boxes[i]
                label = str(LABELS[class_ids[i]])
                color = colors[1]
                cv2.rectangle(detection_img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
                cv2.putText(detection_img, text, (x, y + 30), font, 3, color, 3)

        #------------------- IDENTIFY OBJECT WE WANT TO GO FOR -------------------#
        if len(resultant)>0:
            focusedIndex = resultant.index(min(resultant)) #this returns the index of the object that we want to focus
            focusX, focusY = posVal[focusedIndex] #get the centre co-ords for the focused object
            rho, phi = cart2pol(focusX,focusY) #convert to polar co-ords

            #draw a line on the image from the centre of the frame to the centre of the focused box
            cv2.line(focus, (focusX+640,focusY+360), (640,360), (0,0,0)) 

            #draw the focused box on the image
            x, y, w, h = boxes[focusedIndex]
            label = str(LABELS[class_ids[focusedIndex]])
            color = colors[1]
            cv2.rectangle(focus, (x, y), (x + w, y + h), color, 2)
            #print('[CALC] Polar Co-ords of focused object is {:.6f} < {:.6f}'.format(rho, phi*180/pi))

            # We want to know the depth of the object, lets find trhe average depth of all depth readings within the detection box, this should be pretty reliable.
            try:
                temp = aligned_depth_frame.get_distance(x,y)
                pass
            except RuntimeError as err:
                print(err)
                pass
                    
            if temp is not 0.00:
                dist = temp
                xcart, ycart = deproject_pixel_to_point((x,y), dist)
                print('[INFO]: Object is located at [{:.3f}, {:.3f}, {:.3f}].'.format(xcart, ycart, dist))


            
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
            cv2.rectangle(recmask, (x,y), (x + w + 10, y + h + 10), (255, 255, 255), -1)
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

                #cmd = "3 {:.3f}#".format(angle)
                #cmd = cmd.encode(encoding='UTF-8')
                #p.stdin.write(cmd)
                #p.stdin.flush()
          
            end = time.time()

            print("[INFO]: Program took {:.6f} seconds".format(end - start))
            print('\n -----------------------------\n')


        merge_top = cv2.hconcat((color_image, detection_img))
        merge_bottom = cv2.hconcat((focus, contour))
        merge = cv2.vconcat((merge_top, merge_bottom))

        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Result', 1280,960)
        cv2.imshow("Result", merge)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.namedWindow('Stream', cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Stream", depth_colormap)

        #cv2.namedWindow('Detections', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('Detections', detection_img)

        cv2.waitKey(1)


            
        


except KeyboardInterrupt:
    # Stop streaming
    print("\n[INFO]: Closing...")
    pipeline.stop()
    cv2.destroyAllWindows()
