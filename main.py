import numpy as np
import argparse
import cv2 as cv
import imutils
import psycopg2

__author__ = 'Lazareva O.A. 4481'
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", required=True, type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "boat", "bus", "car",
           "horse", "motorbike", "person", "pottedplant", "sheep", "train"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


capture = cv.VideoCapture("jakarta.mp4")

# Object detection from Stable camera

object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=7)

# load our serialized model from disk
print("Author: ", __author__)
print("[INFO] loading model...")
net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# database connect
connect = psycopg2.connect(dbname="postgres", user="postgres", password="postgres", host="localhost")
cursor = connect.cursor()

while True:
    ret, frame = capture.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = imutils.resize(frame, width=800)

    height, width, _ = frame.shape

    roi = frame[150: 400, 80: 700]

    blob = cv.dnn.blobFromImage(roi, 0.007843, (250, 700), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area > 1500:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow("Cropped image", roi)
    cv.imshow("Frame", frame)
    cv.imshow("Mask", mask)

    key = cv.waitKey(30)
    if key == 27:
        break

capture.release()
cv.destroyAllWindows()

