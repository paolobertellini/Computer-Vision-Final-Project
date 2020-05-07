import cv2
import numpy as np
from matplotlib import pyplot as plt

from perspectiveRectification import perspectiveRectification
from circleDetection import circleDetection


def paintingDetection(original):

    frame = cv2.erode(original, None, iterations=2)
    # frame = cv2.dilate(frame, None, iterations=2)

    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 11)
    # frame = cv2.equalizeHist(frame)
    frame = cv2.medianBlur(frame, 13)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    Hl = cv2.getTrackbarPos("Hl", "Mask trackbars")
    Sl = cv2.getTrackbarPos("Sl", "Mask trackbars")
    Vl = cv2.getTrackbarPos("Vl", "Mask trackbars")
    Hg = cv2.getTrackbarPos("Hg", "Mask trackbars")
    Sg = cv2.getTrackbarPos("Sg", "Mask trackbars")
    Vg = cv2.getTrackbarPos("Vg", "Mask trackbars")

    mask = cv2.inRange(frame_hsv, (Vl, Sl, Hl), (Vg, Sg, Hg))
    # mask = cv2.medianBlur(mask, 21)
    # mask = 255 - mask

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = None

    for cnt in contours:

        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:

            if cv2.contourArea(approx) > 20000:

                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

                rect_frame, out = perspectiveRectification(original, approx)

                plt.subplot(221), plt.imshow(original[:, :, ::-1]), plt.title('Input')
                if np.size(out) != 0:
                    plt.subplot(224), plt.imshow(out[:, :, ::-1]), plt.title('Output')
                frame_pr = original.copy()
                cv2.drawContours(frame_pr, [approx], 0, (0, 255, 0), 3)
                plt.subplot(222), plt.imshow(frame_pr[:, :, ::-1]), plt.title('Contour')
                plt.subplot(223), plt.imshow(rect_frame[:, :, ::-1]), plt.title('Warp')
                plt.show()

    frame, edges = circleDetection(original, frame)

    return frame, mask, edges, out
