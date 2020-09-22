import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import time
import os



pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("cfg/yolov3.cfg", "weights/yolov3.weights")

classes = []

with open("data/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3)) #make a colour for each class

# Start streaming
pipeline.start(config)

try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        (h, w, c) = color_image.shape #get height width and channels

        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (416, 416)), 0.007843, (416, 416), 127.5) #make a blob

        net.setInput(blob)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        outs = net.forward(output_layers)


        class_ids = []
        confidences = []
        boxes = []
        # loop over the detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.7:
                #object detected!!!!
                    center_x = int(detection[0]*w)
                    center_y = int(detection[1]*h)
                    w = int(detection[2] * w)
                    h = int(detection[3] * h)

                     # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(color_image, label, (x, y + 30), font, 3, color, 3)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
    