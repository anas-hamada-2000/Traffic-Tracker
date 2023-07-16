import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/Road traffic video for object recognition.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("Untitled design (1).png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

left_limits = [170, 450, 570, 450]
right_limits = [700, 450, 1080, 450]
totalCount_left = []
totalCount_right = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphicsLeft = cv2.imread("graphics_left.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphicsLeft, (0, 0))
    imgGraphicsRight = cv2.imread("graphics_right.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphicsRight, (823, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (left_limits[0], left_limits[1]), (left_limits[2], left_limits[3]), (0, 0, 255), 5)
    cv2.line(img, (right_limits[0], right_limits[1]), (right_limits[2], right_limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if left_limits[0] < cx < left_limits[2] and left_limits[1] - 15 < cy < left_limits[1] + 15:
            if totalCount_left.count(id) == 0:
                totalCount_left.append(id)
                cv2.line(img, (left_limits[0], left_limits[1]), (left_limits[2], left_limits[3]), (0, 255, 0), 5)
        if right_limits[0] < cx < right_limits[2] and right_limits[1] - 15 < cy < right_limits[1] + 15:
            if totalCount_right.count(id) == 0:
                totalCount_right.append(id)
                cv2.line(img, (right_limits[0], right_limits[1]), (right_limits[2], right_limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(totalCount_left)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.putText(img, str(len(totalCount_right)), (950, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(1) == ord('q'):
        break