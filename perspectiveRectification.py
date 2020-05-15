import math

import cv2
import numpy as np


def adjustBox(box):
    p0 = np.sort(np.sort(box, order='x')[:2], order='y')[0]
    p1 = np.sort(np.sort(box, order='x')[2:], order='y')[0]
    p2 = np.sort(np.sort(box, order='x')[2:], order='y')[-1]
    p3 = np.sort(np.sort(box, order='x')[:2], order='y')[-1]
    return np.float32([[p0['x'], p0['y']], [p1['x'], p1['y']], [p2['x'], p2['y']], [p3['x'], p3['y']]])


def perspectiveRectification(frame, approx):
    dtype = [('x', int), ('y', int)]
    values = [(approx[0][0][0], approx[0][0][1]), (approx[1][0][0], approx[1][0][1]),
              (approx[2][0][0], approx[2][0][1]), (approx[3][0][0], approx[3][0][1])]
    box = np.array(values, dtype)
    box = adjustBox(box)

    cardH = math.sqrt((box[2][0] - box[1][0]) * (box[2][0] - box[1][0]) +
                      (box[2][1] - box[1][1]) * (box[2][1] - box[1][1]))
    # cardW = ratio * cardH;
    cardW = math.sqrt((box[0][0] - box[1][0]) * (box[0][0] - box[1][0]) +
                      (box[0][1] - box[1][1]) * (box[0][1] - box[1][1]))

    rect_box = np.float32([[box[0][0], box[0][1]], [box[0][0] + cardW, box[0][1]],
                           [box[0][0] + cardW, box[0][1] + cardH], [box[0][0], box[0][1] + cardH]])

    M = cv2.getPerspectiveTransform(box, rect_box)

    offsetSize = 2000
    transformed = np.zeros((int(cardW + offsetSize), int(cardH + offsetSize)), dtype=np.uint8);

    rect_frame = cv2.warpPerspective(frame, M, transformed.shape)
    out = rect_frame[int(rect_box[0][1]):int(rect_box[2][1]), int(rect_box[0][0]):int(rect_box[1][0])]

    return rect_frame, out
