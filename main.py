import numpy as np

from scipy import signal
import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture('VIRB0400.MP4')

while(cap.isOpened()):
    ret, frame = cap.read()
    edges = cv2.Canny(frame, 70, 300)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200)
    for r, theta in lines[0]:
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
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # All the changes made in the input image are finally
    # written on a new image houghlines.jpg
    cv2.imwrite('linesDetected.jpg', frame)
    cv2.imshow('mask', edges)

    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindowsows()