import cv2
import os
import numpy as np
from paintingDetection import paintingDetection

paolo_path = 'C:/VCS-project/VCS-project/Project material/videos/all/'
dav_path = '/media/davide/aukey/progetto_vision/videos/'
pepp_path = '/media/peppepc/Volume/Peppe/Unimore/Vision and Cognitive Systems/Project material/videos/'

path = paolo_path
videos = os.listdir(path)
def nothing(x): pass

cv2.namedWindow("Mask trackbars")
cv2.moveWindow("Mask trackbars", 40, 30)
cv2.resizeWindow("Mask trackbars", 300, 250)
cv2.createTrackbar("Hl", "Mask trackbars", 0, 180, nothing)
cv2.createTrackbar("Sl", "Mask trackbars", 0, 255, nothing)
cv2.createTrackbar("Vl", "Mask trackbars", 0, 255, nothing)
cv2.createTrackbar("Hg", "Mask trackbars", 83, 180, nothing)
cv2.createTrackbar("Sg", "Mask trackbars", 255, 255, nothing)
cv2.createTrackbar("Vg", "Mask trackbars", 255, 255, nothing)

cv2.namedWindow("Circles trackbars")
cv2.moveWindow("Circles trackbars", 40, 350)
cv2.resizeWindow("Circles trackbars", 300, 500)
cv2.createTrackbar("k1", "Circles trackbars", 1, 100, nothing)
cv2.createTrackbar("dist_centri", "Circles trackbars", 1000, 2000, nothing)
cv2.createTrackbar("p1", "Circles trackbars", 50, 200, nothing)
cv2.createTrackbar("false_circle", "Circles trackbars", 70, 200, nothing)
cv2.createTrackbar("minR", "Circles trackbars", 100, 1000, nothing)
cv2.createTrackbar("maxR", "Circles trackbars", 1000, 1500, nothing)
cv2.createTrackbar("th1", "Circles trackbars", 100, 300, nothing)
cv2.createTrackbar("th2", "Circles trackbars", 200, 500, nothing)

for videoFile in videos:

    video = cv2.VideoCapture(path + videoFile)
    frame_counter = 0

    while video.isOpened():

        ret, frame = video.read()

        if frame is not None:

            frame_counter += 1

            if frame_counter % 50 == 0:

                frame, mask, edges, out = paintingDetection(frame)

                # output
                cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
                cv2.moveWindow("Detections", 370, 350)
                cv2.resizeWindow("Detections", 600, 350)
                cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
                cv2.moveWindow("Mask", 800, 30)
                cv2.resizeWindow("Mask", 400, 250)
                cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
                cv2.moveWindow("Edges", 370, 30)
                cv2.resizeWindow("Edges", 400, 250)
                cv2.namedWindow('Painting', cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow("Painting", 1000, 350)


                cv2.imshow('Detections', frame)
                cv2.imshow('Mask', mask)
                cv2.imshow('Edges', edges)
                if out is not None and np.size(out) != 0:
                    # cv2.resizeWindow("Painting", out.shape[1] // 5, out.shape[0] // 5)
                    cv2.imshow('Painting', out)

        # stop
        if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT)-1:
            video.release()
            break

video.release()
cv2.destroyAllWindows()