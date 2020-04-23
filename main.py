import cv2
import os
from paintingDetection import paintingDetection

paolo_path = 'C:/VCS-project/VCS-project/Project material/videos/all/'
path = paolo_path
videos = os.listdir(path)

# canny trackbars
def nothing(x): pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Th1", "Trackbars", 50, 500, nothing)
cv2.createTrackbar("Th2", "Trackbars", 30, 500, nothing)

for videoFile in videos:

    video = cv2.VideoCapture(path + videoFile)
    frame_counter = 0

    while video.isOpened():

        ret, frame = video.read()

        if frame is not None:
            frame_counter += 1
            edges, frame = paintingDetection(frame, video)

            # -- output --
            cv2.namedWindow('input', cv2.WINDOW_NORMAL)
            cv2.namedWindow('edges', cv2.WINDOW_NORMAL)

            cv2.imshow('input', frame)
            cv2.imshow('edges', edges)

        # stop
        if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT)-1:
            video.release()
            break

    video.release()
cv2.destroyAllWindows()