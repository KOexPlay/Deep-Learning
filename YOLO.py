from ultralytics import YOLO
import cv2
import sys
import numpy as np


YOLO8data = r"C:\Users\kaige\python projects\Deep Learning\yolov8n.pt"



model = YOLO(YOLO8data)

webcam = cv2.VideoCapture(0)
while True:
    (_, im) = webcam.read()
    result = model(im)
    YOLObox = result[0].plot()
    cv2.imshow("YOLO8", YOLObox)
    key = cv2.waitKey(1)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()


