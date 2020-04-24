import cv2
import os
from paintingDetection import paintingDetection
from paintingDetectionAdaThres import paintingDetectionAdaThres

paolo_path = 'C:/VCS-project/VCS-project/Project material/videos/all/'
dav_path = '/media/davide/aukey/progetto_vision/videos/'
pepp_path = '/media/giuseppe/Volume/Peppe/Unimore/Vision and Cognitive Systems/Project material/videos/'

path = pepp_path
videos = os.listdir(path)

# trackbars
def nothing(x): pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Hl", "Trackbars", 0, 400, nothing)
cv2.createTrackbar("Sl", "Trackbars", 0, 400, nothing)
cv2.createTrackbar("Vl", "Trackbars", 0, 400, nothing)
cv2.createTrackbar("Hg", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Sg", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Vg", "Trackbars", 255, 255, nothing)


for videoFile in videos:

    video = cv2.VideoCapture(path + videoFile)
    frame_counter = 0

    while video.isOpened():

        ret, frame = video.read()

        if frame is not None:
            frame_counter += 1

            #frame, thr = paintingDetection(frame)
            frame, thr = paintingDetectionAdaThres(frame)
            # -- output --
            cv2.namedWindow('input', cv2.WINDOW_NORMAL)
            cv2.namedWindow('edges', cv2.WINDOW_NORMAL)

            cv2.imshow('input', frame)
            cv2.imshow('edges', thr)

        # stop
        if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT)-1:
            video.release()
            break

    video.release()
cv2.destroyAllWindows()