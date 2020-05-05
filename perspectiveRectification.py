import math

import cv2
import numpy as np


def adjustBox(box):
    p0 = np.sort(np.sort(box, order='x')[:2], order='y')[0]
    p1 = np.sort(np.sort(box, order='x')[2:], order='y')[0]
    p2 = np.sort(np.sort(box, order='x')[2:], order='y')[-1]
    p3 = np.sort(np.sort(box, order='x')[:2], order='y')[-1]

    x = np.abs(np.sort(box, order='x')[-1][0] - np.sort(box, order='x')[0][0])
    y = np.abs(np.sort(box, order='y')[-1][0] - np.sort(box, order='y')[0][0])
    ratio = y / x

    return np.float32([[p0['x'], p0['y']], [p1['x'], p1['y']], [p2['x'], p2['y']], [p3['x'], p3['y']]]), ratio


def perspectiveRectification(frame, approx):
    dtype = [('x', int), ('y', int)]
    values = [(approx[0][0][0], approx[0][0][1]), (approx[1][0][0], approx[1][0][1]),
              (approx[2][0][0], approx[2][0][1]), (approx[3][0][0], approx[3][0][1])]
    box = np.array(values, dtype)

    print('UNSORTED', box)
    box, ratio = adjustBox(box)
    print(box)
    # input("Press Enter to continue...")

    # rows, cols, ch = box.shape
    cardH = math.sqrt((box[2][0] - box[1][0]) * (box[2][0] - box[1][0]) +
                      (box[2][1] - box[1][1]) * (box[2][1] - box[1][1]))
    cardW = ratio * cardH;
    rect_box = np.float32([[box[0][0], box[0][1]], [box[0][0] + cardW, box[0][1]],
                           [box[0][0] + cardW, box[0][1] + cardH], [box[0][0], box[0][1] + cardH]])

    M = cv2.getPerspectiveTransform(box, rect_box)

    offsetSize = 2000
    transformed = np.zeros((int(cardW + offsetSize), int(cardH + offsetSize)), dtype=np.uint8);

    rect_frame = cv2.warpPerspective(frame, M, transformed.shape)
    out = rect_frame[int(rect_box[0][1]):int(rect_box[2][1]), int(rect_box[0][0]):int(rect_box[1][0])]

    return rect_frame, out


# test with a single image

image = cv2.imread("test3.jpg")
frame = cv2.erode(image, None, iterations=2)
frame = cv2.medianBlur(frame, 13)
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(frame_hsv, (0, 0, 83), (255, 255, 180))
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        if cv2.contourArea(approx) > 5000:
            perspectiveRectification(image, approx)
