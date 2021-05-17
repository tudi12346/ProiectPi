import math
from math import atan2

import cv2
import numpy as np
from sklearn.metrics import pairwise


def mass_center(img):
    (height, width) = img.shape[:2]
    area = 0.0

    r = c = 0.0
    for i in range(height):
        for j in range(width):
            if img[i, j] == 255:
                area = area + 1
                r = r + i
                c = c + j
    r /= area
    c /= area
    return int(r), int(c)


def hand_detection():
    path_to_image = "2.jpg"

    img = cv2.imread(path_to_image)
    img = cv2.resize(img, (540, 540))

    cv2.imshow('Image original', img)

    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinRegionHSV = cv2.erode(skinRegionHSV, kernel, iterations=2)
    skinRegionHSV = cv2.dilate(skinRegionHSV, kernel, iterations=2)
    skinRegionHSV = cv2.GaussianBlur(skinRegionHSV, (3, 3), 0)

    cv2.imshow("test Tudor", skinRegionHSV)

    contours, hierarchy = cv2.findContours(skinRegionHSV, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    # cv2.drawContours(img, [contours], -1, 127, 2)

    M = cv2.moments(skinRegionHSV)
    r = int(M["m10"] / M["m00"])
    c = int(M["m01"] / M["m00"])
    cv2.circle(skinRegionHSV, (r, c), 5, 0, 50)

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

    print(points)
    print(point1)
    print(point2)
    pp = np.array(points)
    print(pp)
    pp = pp.reshape((-1, 1, 2))
    cv2.fillPoly(skinRegionHSV, [pp], 0)

    cv2.imshow("Center2dd", skinRegionHSV)

    rotate_image = skinRegionHSV
    (h, w) = rotate_image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    if point1[0] != point2[0]:
        radian = atan2(point1[1] - point2[1], point1[0] - point2[0]) + math.pi
        angle = radian * 180 / math.pi
        print(angle)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotate_image = cv2.warpAffine(rotate_image, M, (w, h))

    cv2.imshow("Rotated by 45 Degrees", rotate_image)

    cv2.waitKey(0)
    #
    # cv2.waitKey(0)
    # path_to_image = "4.jpg"
    #
    # img = cv2.imread(path_to_image)
    # img = cv2.resize(img, (540, 540))
    #
    # cv2.imshow('Image', img)
    # hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array([0, 48, 80], dtype="uint8")
    # upper = np.array([20, 255, 255], dtype="uint8")
    #
    # skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # skinRegionHSV = cv2.erode(skinRegionHSV, kernel, iterations=2)
    # skinRegionHSV = cv2.dilate(skinRegionHSV, kernel, iterations=2)
    # skinRegionHSV = cv2.GaussianBlur(skinRegionHSV, (3, 3), 0)
    # thresh1 = skinRegionHSV
    # cv2.imshow("test Tudor", skinRegionHSV)
    #
    #
    #
    # contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # drawing = np.zeros(img.shape, np.uint8)
    # max_area = 0
    # print(len(contours))
    # for i in range(len(contours)):
    #     cnt = contours[i]
    #     area = cv2.contourArea(cnt)
    #     if (area > max_area):
    #         max_area = area
    #         ci = i
    #
    # cnt = contours[ci]
    #
    # print(type(cnt))
    # hull = cv2.convexHull(cnt)
    # moments = cv2.moments(cnt)
    # if moments['m00'] != 0:
    #     cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
    #     cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    #
    # centr = (cx, cy)
    # cv2.circle(img, centr, 5, [0, 0, 255], 2)
    # cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
    # cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)
    #
    # cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    # hull = cv2.convexHull(cnt, returnPoints=False)
    # if (1):
    #     defects = cv2.convexityDefects(cnt, hull)
    #     # print(defects[2])
    #     mind = 0
    #     maxd = 0
    #     for i in range(defects.shape[0]):
    #         s, e, f, d = defects[i, 0]
    #         start = tuple(cnt[s][0])
    #         end = tuple(cnt[e][0])
    #         far = tuple(cnt[f][0])
    #         dist = cv2.pointPolygonTest(cnt, centr, True)
    #         cv2.line(drawing, start, end, [0, 255, 0], 2)
    #
    #         cv2.circle(drawing, far, 5, [0, 0, 255], -1)
    #     print(i)
    #     i = 0
    # cv2.imshow('hull', drawing)
    #
    # cv2.waitKey(0)


if __name__ == '__main__':
    hand_detection()
