import numpy as np

from scipy import signal
import cv2
import math
def nothing(x):
    pass

cap = cv2.VideoCapture('/home/davide/PycharmProjects/HSVedge/VIRB0400.MP4')
cv2.namedWindow("Trackbars")

cv2.createTrackbar("Th1", "Trackbars", 0, 500, nothing)
cv2.createTrackbar("Th2", "Trackbars", 0, 500, nothing)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.medianBlur(frame, ksize=9)
    Z = frame.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    Th1 = cv2.getTrackbarPos("Th1", "Trackbars")
    Th2 = cv2.getTrackbarPos("Th2", "Trackbars")
    edges = cv2.Canny(res2, Th1, Th2)
    contours, hierarchy = cv2.findContours(edges,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    lines= cv2.HoughLinesP(edges, rho=1, theta=math.pi / 180, threshold=70, minLineLength=100, maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2=line[0]
        #cv2.line(frame,(x1,y1),(x2,y2), (0, 0, 255),5)
    '''for r, theta in lines[0]:
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)'''

        # All the changes made in the input image are finally
    # written on a new image houghlines.jpg
    cv2.imwrite('linesDetected.jpg', frame)
    cv2.imshow('mask', res2)
    cv2.imshow('mask1', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindowsows()