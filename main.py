import cv2
import os
from paintingDetection import paintingDetection

paolo_path = 'C:/VCS-project/VCS-project/Project material/videos/all/'
dav_path = '/media/davide/aukey/progetto_vision/videos/'

path = paolo_path
videos = os.listdir(path)
def nothing(x): pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Hl", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("Sl", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Vl", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Hg", "Trackbars", 83, 180, nothing)
cv2.createTrackbar("Sg", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Vg", "Trackbars", 255, 255, nothing)

for videoFile in videos:

    video = cv2.VideoCapture(path + videoFile)
    frame_counter = 0

    while video.isOpened():

        ret, frame = video.read()

        if frame is not None:

            frame_counter += 1

            if frame_counter % 100 == 0:

                print(frame_counter)

                frame, mask = paintingDetection(frame)

                # output
                cv2.namedWindow('input', cv2.WINDOW_NORMAL)
                cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
                cv2.imshow('input', frame)
                cv2.imshow('mask', mask)

        # stop
        if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT)-1:
            video.release()
            break

video.release()
cv2.destroyAllWindows()