import cv2
import numpy as np

def paintingDetectionAdaThres(frame):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv2.filter2D(frame, cv2.CV_32F, kernel)
    sharp = np.float32(frame)
    imgResult = sharp - imgLaplacian
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')

    blur = cv2.medianBlur(imgResult, 15)
    frame2 = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    thr = cv2.adaptiveThreshold(frame2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 2)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) ==4:
            if cv2.contourArea(approx) > 10000:
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

    return frame, thr