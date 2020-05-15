import cv2
import time
from perspectiveRectification import perspectiveRectification
from paintingRetrieval import retrieval
import matplotlib.pyplot as plt
import numpy as np

p = 60
b = p - 5
save = False

def cut(bordered, bbox):
    x, y, w, h = bbox
    return bordered[y:y + h + 2 * b, x:x + w + 2 * b]

def detection(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.erode(frame_hsv, None, iterations=5)
    frame_hsv = cv2.medianBlur(frame_hsv, 17)

    mask = cv2.inRange(frame_hsv, (0, 0, 80), (255, 255, 255))
    mask = 255 - mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for i, cnt in enumerate(contours):

        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if cv2.contourArea(approx) > 20000:
            if len(approx) == 4:
                detections.append(approx)

    return detections


def boundingBox(frame):

    detections = []

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.erode(frame_hsv, None, iterations=5)
    frame_hsv = cv2.medianBlur(frame_hsv, 17)

    mask = cv2.inRange(frame_hsv, (0, 0, 80), (255, 255, 255))
    mask = 255 - mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blue = frame.copy()
    bordered = cv2.copyMakeBorder(frame, p, p, p, p, borderType=cv2.BORDER_CONSTANT)

    for i, cnt in enumerate(contours):

        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if cv2.contourArea(approx) > 20000:
            bbox = cv2.boundingRect(approx)
            x, y, w, h = bbox
            cv2.rectangle(blue, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cutted = cut(bordered, bbox)

            if save:
                cv2.imwrite('detect-' + str(time.time()) + ".jpg", cutted)

            # plt.imshow(cutted[:, :, ::-1])
            # plt.show()

            detect = {"cut":cutted, "bbox":bbox}
            detections.append(detect)

    return blue, detections



def segmentation(bb, paintings_descriptors):

    green = bb.copy()
    img = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(img, 17)
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    th3 = 255 - th3

    contours, _ = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = None

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            if cv2.contourArea(approx) > 10000:


                rect, out = perspectiveRectification(bb, approx)
                color = (0, 0, 255)
                if out is not None and np.size(out) != 0:
                    ret = retrieval(bb, paintings_descriptors)
                    plt.subplot(133), plt.xlabel(ret)
                    plt.subplot(133), plt.imshow(out[:, :, ::-1])
                    if ret != 'soreta':
                        color = (0, 255, 0)

                cv2.drawContours(green, [approx], 0, color, 3)
                plt.subplot(131), plt.imshow(green[:, :, ::-1])
                plt.subplot(132), plt.imshow(th3, cmap='gray')

                plt.show()