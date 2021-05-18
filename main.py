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


def rotate_point(point_original, point_center, angle):
    point = [0, 0]
    point[0] = int(((point_original[0] - point_center[0]) * math.cos(angle)) - ((point_original[1] - point_center[0])
                                                                                * math.sin(angle)) + point_center[0])
    point[1] = int(((point_original[0] - point_center[0]) * math.sin(angle)) + ((point_original[1] - point_center[0]) *
                                                                                math.cos(angle)) + point_center[1])
    return point


def hand_detection():
    type_image = '3'
    path_to_image = f'database Non Procees/Test {type_image}.jpg'

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
    skinRegionHSV = cv2.threshold(skinRegionHSV, 200, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Binary Image Hand", skinRegionHSV)

    contours, hierarchy = cv2.findContours(skinRegionHSV, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))

    M = cv2.moments(skinRegionHSV)
    r = int(M["m10"] / M["m00"])
    c = int(M["m01"] / M["m00"])

    convex = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, convex)

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

    pp = np.array(points)
    pp = pp.reshape((-1, 1, 2))

    cv2.fillPoly(skinRegionHSV, [pp], 0)

    cv2.imshow("Clip Hand Image", skinRegionHSV)

    rotate_image = skinRegionHSV
    (h, w) = rotate_image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    if point1[0] != point2[0]:
        radian = atan2(point1[1] - point2[1], point1[0] - point2[0]) + math.pi
        angle = radian * 180 / math.pi
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotate_image[point1[0]][point1[1]] = 128
        rotate_image[point2[0]][point2[1]] = 128
        y_val = 9999999
        for i in range(h):
            for j in range(w):
                if rotate_image[i][j] != 0 and rotate_image[i][j] != 255:
                    if j < y_val:
                        y_val = j
        rotate_image = cv2.warpAffine(rotate_image, M, (w, h))

    cv2.imshow("Rotated Hand Image", rotate_image)
    rotate_image = rotate_image[:y_val, ]
    cv2.imshow("Resize Hand Image", rotate_image)

    directory = 'database/'
    cv2.imwrite(directory + type_image + '.jpg', rotate_image)

    cv2.waitKey(0)


if __name__ == '__main__':
    hand_detection()
