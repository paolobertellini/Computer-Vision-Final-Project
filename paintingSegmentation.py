import cv2
import numpy as np


def orderPoints(points):
    point = [('x', int), ('y', int)]
    values = [(points[0][0][0], points[0][0][1]),
              (points[1][0][0], points[1][0][1]),
              (points[2][0][0], points[2][0][1]),
              (points[3][0][0], points[3][0][1])]
    box = np.array(values, point)
    p0 = np.sort(np.sort(box, order='x')[:2], order='y')[0]
    p1 = np.sort(np.sort(box, order='x')[2:], order='y')[0]
    p2 = np.sort(np.sort(box, order='x')[2:], order='y')[-1]
    p3 = np.sort(np.sort(box, order='x')[:2], order='y')[-1]
    return np.int32([[[p0['x'], p0['y']]], [[p1['x'], p1['y']]], [[p2['x'], p2['y']]], [[p3['x'], p3['y']]]])


def paintingSegmentation(cutted):
    gray = cv2.cvtColor(cutted, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, 2)
    th = 255 - th

    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = np.size(cutted)
    best = None

    segmentation = cutted.copy()

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            cv2.drawContours(segmentation, [approx], 0, (255, 255, 0), 3)
            if area > (np.size(cutted) / 3) / 10:
                if area < min_area:
                    best = orderPoints(approx)
                    min_area = area

    # if best is not None:
    #     cv2.drawContours(segmentation, [best], 0, (0, 255, 0), 3)
    #     plt.subplot(121), plt.imshow(segmentation[:, :, ::-1])
    #     plt.subplot(122), plt.imshow(th, cmap='gray')
    #     plt.show()

    return best
