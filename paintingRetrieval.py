import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from perspectiveRectification import perspectiveRectification

pepp_path = '/media/peppepc/Volume/Peppe/Unimore/Vision and Cognitive Systems/Project material/paintings_db'

path = pepp_path
paintings = os.listdir(path)
output = open('paintingsKP.txt', "w")

def paintingRetrieval(rectPaint):
    rectPaint= cv2.cvtColor(rectPaint, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    paintKP, des1 = orb.detectAndCompute(rectPaint, None)
    print(paintKP)
    #
    # for paint in paintings:
    #     orb = cv2.ORB_create()
    #     paintKPs, des = orb.detectAndCompute(paint, None)
    #
    #     features = [str(f) for f in paintKPs]
    #     output.write(features)
    img = cv2.drawKeypoints(rectPaint, paintKP, rectPaint, color=(0,255,0), flags=0)

    return img

image = cv2.imread("test6.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

frame = cv2.erode(image, None, iterations=2)
frame = cv2.medianBlur(frame, 13)
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(frame_hsv, (0, 0, 83), (255, 255, 180))
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        if cv2.contourArea(approx) > 5000:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

            rect_frame, out = perspectiveRectification(image, approx)
            kp_detected = paintingRetrieval(out)

plt.subplot(221), plt.imshow(image), plt.title('Input')
plt.subplot(222), plt.imshow(kp_detected), plt.title('Output')
plt.show()
