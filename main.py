import cv2
import os
from paintingDetection import paintingDetection

paolo_path = 'C:/VCS-project/VCS-project/Project material/videos/all/'
dav_path = '/media/davide/aukey/progetto_vision/videos/'

path = paolo_path
videos = os.listdir(path)
def nothing(x): pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Hl", "Trackbars", 16, 400, nothing)
cv2.createTrackbar("Sl", "Trackbars", 40, 400, nothing)
cv2.createTrackbar("Vl", "Trackbars", 88, 400, nothing)
cv2.createTrackbar("Hg", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Sg", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Vg", "Trackbars", 255, 255, nothing)

cv2.namedWindow("Circles")
k1 = cv2.createTrackbar("k1", "Circles", 1, 100, nothing)
dist_centri = cv2.createTrackbar("dist_centri", "Circles", 40, 200, nothing)
p1 = cv2.createTrackbar("p1", "Circles", 50, 200, nothing)
false_circle = cv2.createTrackbar("false_circle", "Circles", 50, 200, nothing)
minR = cv2.createTrackbar("minR", "Circles", 300, 1000, nothing)
maxR = cv2.createTrackbar("maxR", "Circles", 500, 1000, nothing)
th1 = cv2.createTrackbar("th1", "Circles", 1, 100, nothing)
th2 = cv2.createTrackbar("th2", "Circles", 1, 100, nothing)

for videoFile in videos:

    video = cv2.VideoCapture(path + videoFile)
    frame_counter = 0

    while video.isOpened():

        ret, frame = video.read()

        if frame is not None:
            frame_counter += 1

            frame, mask = paintingDetection(frame)

            # -- output --
            cv2.namedWindow('input', cv2.WINDOW_NORMAL)
            cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('edges', cv2.WINDOW_NORMAL)

            cv2.imshow('input', frame)
            cv2.imshow('mask', mask)
            # cv2.imshow('edges', edges)

        # stop
        if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT)-1:
            video.release()
            break

video.release()
cv2.destroyAllWindows()