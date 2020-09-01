#Commands: 0 = wait, 1 = scan, 2 = servo, 3  = rotate

#import pyrealsense2 as rs
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


    #get indiv colours for each class
    np.random.seed(42)
    colors = np.random.uniform(0, 255, size=(len(LABELS), 3)) #make a colour for each class

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #set the iamge path, makes it easy for reading img in
    image_path = args["image"]

    if args["image"] is not None: 
        #read img in
        color_image = cv2.imread(image_path)
        color_image = cv2.resize(color_image, (640,480))
        #get img dimensions
        (h, w, c) = color_image.shape

        blob = cv2.dnn.blobFromImage(color_image, 1/255, (416, 416),
                swapRB=True, crop=False) #make a blob

        net.setInput(blob)
        start = time.time()

        outs = net.forward(layer_names)
        end = time.time()

        print("[INFO] YOLO took {:.6f} seconds".format(end - start))


        class_ids = []
        confidences = []
        boxes = []
        resultant = []
        posVal = []

        #------------------- LOOP OVER DETECTIONS -------------------#

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.7:
                #object detected!!!!
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    posVal.append([centerX-320, centerY-240])
                    # Rectangle coordinates
                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)

                    resultant.append(sqrt((centerX-320)**2 + (centerY-240)**2)) 
                    #note ^ opencv pixel co-ords are form top corner, therefore by subtracting 
                    # the middle of the frame we change the reference from the corner to the middle.

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if (i in indexes) and (class_ids[i] == 46): #class id of 46 is banana, could also be LABELS(class_ids[i]) == "banana":
                print(i)
                x, y, w, h = boxes[i]
                label = str(LABELS[class_ids[i]])
                color = colors[i]
                cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])

                cv2.putText(color_image, text, (x, y + 30), font, 3, color, 3)

        #------------------- IDENTIFY BANANA WE WANT TO GO FOR -------------------#

        focus = cv2.imread(image_path)
        focus = cv2.resize(focus, (640, 480))

        focusedIndex = resultant.index(min(resultant)) #this returns the index of the object that we want to focus
        print(focusedIndex)
        focusX, focusY = posVal[focusedIndex] #get the centre co-ords for the focused object
        rho, phi = cart2pol(focusX,focusY) #convert to polar co-ords
        cv2.line(focus, (focusX+320,focusY+240), (320,240), (0,0,0)) #draw a line on the image from the centre of the frame to the centre of the focused box

        x, y, w, h = boxes[focusedIndex]
        label = str(LABELS[class_ids[focusedIndex]])
        color = colors[focusedIndex]
        cv2.rectangle(focus, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[class_ids[focusedIndex]], confidences[focusedIndex])

        cv2.putText(focus, text, (x, y + 30), font, 3, color, 3)

        print(f'[CALC] Polar Co-ords of focused object is {rho} < {phi*180/pi}')

    return(focus, focusX, focusY)

def ObjDetectImg(img, layer_names, net):

    #get indiv colours for each class
    np.random.seed(42)
    colors = np.random.uniform(0, 255, size=(len(LABELS), 3)) #make a colour for each class

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    color_image = img
    #get img dimensions
    (h, w, c) = color_image.shape

    blob = cv2.dnn.blobFromImage(color_image, 1/255, (416, 416),
                swapRB=True, crop=False) #make a blob

    net.setInput(blob)
    start = time.time()

    outs = net.forward(layer_names)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))


    class_ids = []
    confidences = []
    boxes = []
    resultant = []
    posVal = []

    #------------------- LOOP OVER DETECTIONS -------------------#

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7:
            #object detected!!!!
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                posVal.append([centerX-320, centerY-240])
                # Rectangle coordinates
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                resultant.append(sqrt((centerX-320)**2 + (centerY-240)**2)) 
                #note ^ opencv pixel co-ords are form top corner, therefore by subtracting 
                # the middle of the frame we change the reference from the corner to the middle.

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if (i in indexes) and (class_ids[i] == 46): #class id of 46 is banana, could also be LABELS(class_ids[i]) == "banana":
            x, y, w, h = boxes[i]
            label = str(LABELS[class_ids[i]])
            color = colors[i]
            cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])

            cv2.putText(color_image, text, (x, y + 30), font, 3, color, 3)

    #------------------- IDENTIFY BANANA WE WANT TO GO FOR -------------------#
    if resultant is not None:
        focus = img

        focusedIndex = resultant.index(min(resultant)) #this returns the index of the object that we want to focus
        focusX, focusY = posVal[focusedIndex] #get the centre co-ords for the focused object
        rho, phi = cart2pol(focusX,focusY) #convert to polar co-ords
        cv2.line(focus, (focusX+320,focusY+240), (320,240), (0,0,0)) #draw a line on the image from the centre of the frame to the centre of the focused box

        x, y, w, h = boxes[focusedIndex]
        label = str(LABELS[class_ids[focusedIndex]])
        color = colors[focusedIndex]
        cv2.rectangle(focus, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[class_ids[focusedIndex]], confidences[focusedIndex])

        cv2.putText(focus, text, (x, y + 30), font, 3, color, 3)

        print('[CALC] Polar Co-ords of focused object is {:.6f} < {:.6f}'.format(rho, phi*180/pi))

        #cv2.imshow("Focused object", focus)
        #cv2.imshow('Detections', color_image)
        
        return(x, y, w, h, rho)


#-------------------------------------------------#    
#------------------- CONFIG ----------------------#
#-------------------------------------------------#


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image")

args = vars(ap.parse_args())

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

#get all the classes
labelsPath = "data/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

#get individual colours for each class
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(LABELS), 3)) #make a colour for each class

layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#-------------------------------------------------#
#------------------- DETECTION -------------------#
#-------------------------------------------------#


if args["image"] is not None:

    image_path = args["image"]
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640,480))

    x, y, w, h, rho = ObjDetectImg(img, layer_names, net)

    if rho > 50:
        print("[RESULT] Command 1, servo to object...")
        cmd = "2 {x} {y}\n".encode(encoding='UTF-8')
        p.stdin.write(cmd)
        p.stdin.flush()

    else: 
        print("[RESULT] End effector is above object, computing orientation")
        

        #first we want to crop our image to the bounding box of the focused object
        #this will make sure we only get one orientation reading.
        
         #------------------- NOW WE FIND ORIENTATION -------------------#
        start = time.time()

        img = cv2.imread(image_path)
        img = cv2.resize(img, (640,480))
        crop = img[y:y+h, x:x+w]
        # Convert image to hsv
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) 
        #lets get some values for our hsv thresholds, what colours should we play between?

        weaker = np.array([20, 100, 100]) #kind of a pale yellow colour
        stronger = np.array([30, 255, 255]) #strong, vibrant yellow.

        #make a mask

        mask = cv2.inRange(hsv, weaker, stronger) 
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) #mask to bw if using

        for i, c in enumerate(contours):
            # Calculate the area of each contour
            area = cv2.contourArea(c)
            # Ignore contours that are too small or too large
            if area < 1e3 or 1e5 < area:
                continue

            #   Draw each contour only for visualisation purposes
            cv2.drawContours(crop, contours, i, (0, 0, 255), 2)
            # Find the orientation of each shape
            angle = getOrientation(c, crop)
            print("Orientation is", angle*(180/pi))

        cmd = "3 {:.3f}#".format(angle)
        cmd = cmd.encode(encoding='UTF-8')
        p.stdin.write(cmd)
        p.stdin.flush()
        end = time.time()
        print("[INFO] Orientation calc took {:.6f} seconds".format(end - start))
        cv2.imshow("ori", img)
       


    cv2.waitKey()
    cv2.destroyAllWindows()
    p.kill()

else: #if NO image has been parsed, we want to do webcam
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    

    try:
        while True:

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            #depth_frame = frames.get_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            (h, w, c) = color_image.shape #get height width and channels
           

            start = time.time()

            x, y, w, h, rho = ObjDetectImg(color_image, layer_names, net)

            img = color_image
            crop = img[y:y+h, x:x+w]
            
            # Convert image to hsv
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) 
            #lets get some values for our hsv thresholds, what colours should we play between?

            weaker = np.array([20, 100, 100]) #kind of a pale yellow colour
            stronger = np.array([30, 255, 255]) #strong, vibrant yellow.

            #make a mask

            mask = cv2.inRange(hsv, weaker, stronger) 
            _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) #mask to bw if using

            for i, c in enumerate(contours):
                # Calculate the area of each contour
                area = cv2.contourArea(c)
                # Ignore contours that are too small or too large
                if area < 1e3 or 1e5 < area:
                    continue

                #   Draw each contour only for visualisation purposes
                cv2.drawContours(crop, contours, i, (0, 0, 255), 2)
                # Find the orientation of each shape
                #angle = getOrientation(c, crop)
                #print("Orientation is", angle*(180/pi))

            #cmd = "3 {:.3f}#".format(angle)
            #cmd = cmd.encode(encoding='UTF-8')
            #p.stdin.write(cmd)
            #p.stdin.flush()
            end = time.time()
            seconds = start - end
            print("[INFO]: ", seconds, "time taken.")

            #cv2.imshow("ori", img)
            
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)


    except KeyboardInterrupt:
        # Stop streaming
        print("\n [INFO]: Closing...")
        pipeline.stop()
        cv2.destroyAllWindows()
