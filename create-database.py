import math
from math import atan2
import cv2
import numpy as np
from sklearn.metrics import pairwise


def rotate_point(point_original, point_center, angle):
    point = [0, 0]
    point[0] = int(((point_original[0] - point_center[0]) * math.cos(angle)) - ((point_original[1] - point_center[0])
                                                                                * math.sin(angle)) + point_center[0])
    point[1] = int(((point_original[0] - point_center[0]) * math.sin(angle)) + ((point_original[1] - point_center[0]) *
                                                                                math.cos(angle)) + point_center[1])
    return point


def init_hand_detection_database(type_image):
    path_to_image = f'database Non Procees/Test {type_image}.jpg'

    img = cv2.imread(path_to_image)
    img = cv2.resize(img, (540, 540))
    image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Bounds for the skin colour
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    image_skin = cv2.inRange(image_HSV, lower, upper)
    my_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # Eliminate the noise
    image_skin = cv2.erode(image_skin, my_kernel, iterations=2)
    image_skin = cv2.dilate(image_skin, my_kernel, iterations=2)
    image_skin = cv2.GaussianBlur(image_skin, (3, 3), 0)
    image_skin = cv2.threshold(image_skin, 200, 255, cv2.THRESH_BINARY)[1]

    # Get contours
    contours, hierarchy = cv2.findContours(image_skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))

    # Get center
    M = cv2.moments(image_skin)
    r = int(M["m10"] / M["m00"])
    c = int(M["m01"] / M["m00"])

    convex = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, convex)

    # Get points of the rectangle of palm
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
    cv2.fillPoly(image_skin, [pp], 0)


    rotate_image = image_skin
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

    rotate_image = rotate_image[:y_val, ]
    rotate_image = cv2.resize(rotate_image, (128, 128))

    directory = 'database/'
    cv2.imwrite(directory + type_image + '.jpg', rotate_image)


if __name__ == '__main__':
    init_hand_detection_database('1')
    init_hand_detection_database('2')
    init_hand_detection_database('3')
    init_hand_detection_database('4')
    init_hand_detection_database('5')
    init_hand_detection_database('Like')
    init_hand_detection_database('Ok')
    init_hand_detection_database('Pistol')
    init_hand_detection_database('Rock')
