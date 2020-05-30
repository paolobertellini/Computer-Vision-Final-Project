import numpy as np

import os

from peopleDetection import inizializeModel
from pipeline import pipeline
from utils import *

selection = ['GOPR1924.MP4', 'GOPR1940.MP4', '20180206_114604.mp4', 'IMG_9620.MOV', 'IMG_4076.MOV', 'VIRB0413.MP4',
             'IMG_3819.MOV', 'GOPR2039.MP4', 'IMG_2657.MOV',
             'IMG_2657.MOV', 'VIRB0426.MP4', 'GOPR1927.MP4', 'IMG_4075.MOV', 'IMG_4081.MOV', 'IMG_4086.MOV',
             '20180206_112658.mp4']

paolo_path = 'D:/VCS-project/'
dav_path = '/media/davide/aukey/progetto_vision/'
pepp_path = '/media/peppepc/Volume/Peppe/Unimore/Vision and Cognitive Systems/Project material/'

root_path = dav_path

paintings_info = loadPaintingsInfo(root_path)

model = inizializeModel(1)

videos_path = root_path + 'videos/'
videos = os.listdir(videos_path)
videos = np.random.permutation(videos)

for videoFile in videos:
    video = cv2.VideoCapture(videos_path + videoFile)
    print('WATCHING: ' + videoFile)
    frame_counter = 0

    while video.isOpened():
        _, frame = video.read()
        if frame is not None:
            frame_counter += 1
            if frame_counter % 4 == 0:
                pipeline(frame, paintings_info, model)

            # stop
            if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(
                    cv2.CAP_PROP_FRAME_COUNT) - 1:
                video.release()
                break

video.release()
cv2.destroyAllWindows()
