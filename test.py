import cv2
import os
from paintingDetection import paintingDetection


def nothing(x): pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Hl", "Trackbars", 16, 400, nothing)
cv2.createTrackbar("Sl", "Trackbars", 40, 400, nothing)
cv2.createTrackbar("Vl", "Trackbars", 88, 400, nothing)
cv2.createTrackbar("Hg", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Sg", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Vg", "Trackbars", 255, 255, nothing)



while True:
    frame = cv2.imread("test.jpg")
    frame, mask = paintingDetection(frame)

    # -- output --
    cv2.namedWindow('input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('edges', cv2.WINDOW_NORMAL)

    cv2.imshow('input', frame)
    cv2.imshow('mask', mask)
    cv2.waitKey(5)
            # cv2.imshow('edges', edges)

# cv2.destroyAllWindows()