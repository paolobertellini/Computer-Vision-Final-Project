import cv2
import numpy as np
import matplotlib.pyplot as plt

def circleDetection(original, frame):

    k1 = cv2.getTrackbarPos("k1", "Circles trackbars")
    dist_centri = cv2.getTrackbarPos("dist_centri", "Circles trackbars")
    p1 = cv2.getTrackbarPos("p1", "Circles trackbars")
    false_circle = cv2.getTrackbarPos("false_circle", "Circles trackbars")
    minR = cv2.getTrackbarPos("minR", "Circles trackbars")
    maxR = cv2.getTrackbarPos("maxR", "Circles trackbars")
    th1 = cv2.getTrackbarPos("th1", "Circles trackbars")
    th2 = cv2.getTrackbarPos("th2", "Circles trackbars")

    # resized = cv2.resize(frame, (frame.shape[0] // 3, frame.shape[1] // 3), interpolation=cv2.INTER_AREA)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resized = cv2.resize(frame, (320, 240), interpolation =cv2.INTER_AREA)
    edges = cv2.Canny(frame_gray, th1, th2)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, k1, dist_centri,
                                        param1=p1, param2=false_circle, minRadius=minR, maxRadius=maxR)


    if circles is not None:
        detected_circles = np.uint16(np.around(circles))
        for circle in detected_circles[0, :]:
            a, b, r = circle[0], circle[1], circle[2]
            cv2.circle(frame, (a, b), r, (255, 0, 0), 3)

    return frame, edges