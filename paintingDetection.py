import cv2
import numpy as np
from matplotlib import pyplot as plt
from circleDetection import circleDetection


def paintingDetection(frame):

    frame = cv2.erode(frame, None, iterations=2)
    frame = cv2.dilate(frame, None, iterations=2)
    # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 11)
    # frame = cv2.equalizeHist(frame)
    frame = cv2.medianBlur(frame, 13)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


    Hl = cv2.getTrackbarPos("Hl", "Trackbars")
    Sl = cv2.getTrackbarPos("Sl", "Trackbars")
    Vl = cv2.getTrackbarPos("Vl", "Trackbars")
    Hg = cv2.getTrackbarPos("Hg", "Trackbars")
    Sg = cv2.getTrackbarPos("Sg", "Trackbars")
    Vg = cv2.getTrackbarPos("Vg", "Trackbars")

    mask = cv2.inRange(frame_hsv, (Vl, Sl, Hl), (Vg, Sg, Hg))
    # mask = cv2.medianBlur(mask, 21)
    # mask = 255 - mask

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # h = cv2.calcHist(frame, 3, mask, histSize, ranges[, hist[, accumulate]])

    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 4:
            if cv2.contourArea(approx) > 5000:
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

    # cv2.drawContours(frame, approx, -1, (0, 255, 0), 3)

    # edges, frame = circleDetection(frame_gray)



    return frame, mask
