import math
from math import atan2

import cv2
import numpy as np
import os

# Create the directory structure
from sklearn.metrics import pairwise

if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/0")
    os.makedirs("data/train/1")
    os.makedirs("data/train/2")
    os.makedirs("data/train/3")
    os.makedirs("data/train/4")
    os.makedirs("data/train/5")
    os.makedirs("data/test/0")
    os.makedirs("data/test/1")
    os.makedirs("data/test/2")
    os.makedirs("data/test/3")
    os.makedirs("data/test/4")
    os.makedirs("data/test/5")

# Train or test
mode = 'train'
directory = 'data/' + mode + '/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {'zero': len(os.listdir(directory + "/0")),
             'one': len(os.listdir(directory + "/1")),
             'two': len(os.listdir(directory + "/2")),
             'three': len(os.listdir(directory + "/3")),
             'four': len(os.listdir(directory + "/4")),
             'five': len(os.listdir(directory + "/5"))}

    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : " + mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "ZERO : " + str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "ONE : " + str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "TWO : " + str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "THREE : " + str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "FOUR : " + str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "FIVE : " + str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    # Coordinates of the Region of interest
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    cv2.imshow("Frame", frame)

    hsvim = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinRegionHSV = cv2.erode(skinRegionHSV, kernel, iterations=2)
    skinRegionHSV = cv2.dilate(skinRegionHSV, kernel, iterations=2)
    skinRegionHSV = cv2.GaussianBlur(skinRegionHSV, (3, 3), 0)

    roi = cv2.resize(skinRegionHSV, (128, 128))

    contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = max(contours, key=lambda x: cv2.contourArea(x))

        M = cv2.moments(roi)
        r = int(M["m10"] / M["m00"])
        c = int(M["m01"] / M["m00"])
        convex = cv2.convexHull(contours, returnPoints=False)
        defects = cv2.convexityDefects(contours, convex)

        dist = []
        points_start = []
        points_far = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            points_start.append(contours[s][0])
            points_far.append(contours[f][0])

        points = []
        for i in range(len(points_far)):
            ok = True
            for j in range(len(points_start)):
                if pairwise.euclidean_distances([points_far[i]], Y=[points_start[j]])[0][0] < 30:
                    ok = False
            if ok:
                points.append(points_far[i])

        dist_max = 0
        point1 = [0, 0]
        point2 = [0, 0]
        for i in range(len(points)):
            if i == len(points) - 1:
                if pairwise.euclidean_distances([points[i]], Y=[points[0]])[0][0] > dist_max:
                    dist_max = pairwise.euclidean_distances([points[i]], Y=[points[0]])[0][0]
                    point1 = points[i]
                    point2 = points[0]
            else:
                if pairwise.euclidean_distances([points[i]], Y=[points[i + 1]])[0][0] > dist_max:
                    dist_max = pairwise.euclidean_distances([points[i]], Y=[points[i + 1]])[0][0]
                    point1 = points[i]
                    point2 = points[i + 1]

        if points:
            pp = np.array(points)
            pp = pp.reshape((-1, 1, 2))
            cv2.fillPoly(roi, [pp], 0)

            (h, w) = roi.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            if point1[0] != point2[0]:
                radian = atan2(point1[1] - point2[1], point1[0] - point2[0]) + math.pi
                angle = radian * 180 / math.pi
                M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
                roi = cv2.warpAffine(roi, M, (w, h))

    cv2.imshow("ROI", roi)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory + '0/' + str(count['zero']) + '.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory + '1/' + str(count['one']) + '.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory + '2/' + str(count['two']) + '.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory + '3/' + str(count['three']) + '.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory + '4/' + str(count['four']) + '.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory + '5/' + str(count['five']) + '.jpg', roi)

cap.release()
cv2.destroyAllWindows()
