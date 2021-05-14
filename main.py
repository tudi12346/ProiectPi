import cv2
import numpy as np


def mass_center(img):
    (height, width) = img.shape[:2]
    area = 0.0

    r = c = 0.0
    for i in range(height):
        for j in range(width):
            if img[i,j] == 255:
                area = area +1
                r = r + i
                c = c + j
    r /= area
    c /= area
    return int(r), int(c)


def hand_detection():
    path_to_image = "2.jpg"

    img = cv2.imread(path_to_image)
    img = cv2.resize(img, (540, 540))

    cv2.imshow('Image', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('binarized', thresh1)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img.shape, np.uint8)
    max_area = 0
    print (len(contours))
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area > max_area):
            max_area = area
            ci = i


    cnt = contours[ci]

    print(type(cnt))
    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

    centr = (cx, cy)
    cv2.circle(img, centr, 5, [0, 0, 255], 2)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

    cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    hull = cv2.convexHull(cnt, returnPoints=False)
    if (1):
        defects = cv2.convexityDefects(cnt, hull)
        #print(defects[2])
        mind = 0
        maxd = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            dist = cv2.pointPolygonTest(cnt, centr, True)
            cv2.line(drawing, start, end, [0, 255, 0], 2)

            cv2.circle(drawing, far, 5, [0, 0, 255], -1)
        print(i)
        i = 0
    cv2.imshow('hull', drawing)

    cv2.waitKey(0)


if __name__ == '__main__':
    hand_detection()
