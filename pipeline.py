import numpy as np

from paintingDetection import paintingDetection
from paintingRetrieval import paintingRetrieval
from paintingSegmentation import paintingSegmentation
from peopleDetection import peopleDetection
from perspectiveRectification import perspectiveRectification
from utils import *

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.moveWindow("Detection", 10, 20)
cv2.resizeWindow("Detection", 500, 400)

cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
cv2.moveWindow("Segmentation", 10, 480)
cv2.resizeWindow("Segmentation", 300, 200)

cv2.namedWindow('Rectification', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Rectification", 520, 20)

cv2.namedWindow('Retrieval', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Retrieval", 950, 20)

cv2.namedWindow('People detection', cv2.WINDOW_NORMAL)


def pipeline(frame, paintings_info, model):
    framePeople = frame.copy()
    frameSegmented = frame.copy()
    red_frame = np.full(frame.shape, (0, 0, 255), np.uint8)
    frameSegmented = cv2.addWeighted(frameSegmented, 0.4, red_frame, 0.6, 0)

    peopleBoxes, classes = peopleDetection(frame, model, 0.8)
    paintingsBoxes = paintingDetection(frame)

    if (peopleBoxes is not None) & (paintingsBoxes is not None):
        peopleBoxes, paintingsBoxes = removeInnerBox(peopleBoxes, paintingsBoxes)

    if peopleBoxes is not None:
        for x1, y1, x2, y2 in peopleBoxes:
            cv2.rectangle(framePeople, (x1, y1), (x2, y2), (255, 255, 255), 5)

    for x1, y1, x2, y2 in paintingsBoxes:

        c = (0, 0, 255)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        cutted = cut(frame, bbox)
        paintingSegmented = paintingSegmentation(cutted)

        if paintingSegmented is not None:

            cv2.drawContours(frameSegmented, [paintingSegmented + [bbox[0] - 60, bbox[1] - 60]], 0, (0, 255, 0), -1)

            rect_painting = perspectiveRectification(cutted, paintingSegmented)
            if np.size(rect_painting) != 0:
                cv2.imshow("Rectification", resize(400, rect_painting))

            paintingScore, c, info, retrieval = paintingRetrieval(cutted, paintings_info)

            cv2.putText(frame, info, (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Retrieval", resize(400, retrieval))

        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 5)

    cv2.imshow('Detection', frame)
    cv2.imshow('Segmentation', frameSegmented)
    cv2.imshow('People detection', framePeople)
