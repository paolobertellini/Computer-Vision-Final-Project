import math

import cv2
import numpy as np


def perspectiveRectification(frame, segmentationPoints):
    cardH = math.sqrt((segmentationPoints[2][0][0] - segmentationPoints[1][0][0]) * (
            segmentationPoints[2][0][0] - segmentationPoints[1][0][0]) +
                      (segmentationPoints[2][0][1] - segmentationPoints[1][0][1]) * (
                              segmentationPoints[2][0][1] - segmentationPoints[1][0][1]))
    cardW = math.sqrt((segmentationPoints[0][0][0] - segmentationPoints[1][0][0]) * (
            segmentationPoints[0][0][0] - segmentationPoints[1][0][0]) +
                      (segmentationPoints[0][0][1] - segmentationPoints[1][0][1]) * (
                              segmentationPoints[0][0][1] - segmentationPoints[1][0][1]))

    box = np.float32([[segmentationPoints[0][0][0], segmentationPoints[0][0][1]],
                      [segmentationPoints[1][0][0], segmentationPoints[1][0][1]],
                      [segmentationPoints[2][0][0], segmentationPoints[2][0][1]],
                      [segmentationPoints[3][0][0], segmentationPoints[3][0][1]]])

    rect_box = np.float32([[segmentationPoints[0][0][0], segmentationPoints[0][0][1]],
                           [segmentationPoints[0][0][0] + cardW, segmentationPoints[0][0][1]],
                           [segmentationPoints[0][0][0] + cardW, segmentationPoints[0][0][1] + cardH],
                           [segmentationPoints[0][0][0], segmentationPoints[0][0][1] + cardH]])

    M = cv2.getPerspectiveTransform(box, rect_box)

    offsetSize = 2000
    transformed = np.zeros((int(cardW + offsetSize), int(cardH + offsetSize)), dtype=np.uint8);

    rect_frame = cv2.warpPerspective(frame, M, transformed.shape)
    rect_paint = rect_frame[int(rect_box[0][1]):int(rect_box[2][1]), int(rect_box[0][0]):int(rect_box[1][0])]

    return rect_paint
