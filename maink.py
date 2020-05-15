import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from paintingDetection import boundingBox, segmentation
from perspectiveRectification import perspectiveRectification

paolo_path = 'C:/VCS-project/'
dav_path = '/media/davide/aukey/progetto_vision/'
pepp_path = '/media/peppepc/Volume/Peppe/Unimore/Vision and Cognitive Systems/Project material/'

path = paolo_path
paintings_path = path + 'paintings_db/'
video_path = path + 'videos/'
videos = os.listdir(video_path)
videos = np.random.permutation(videos)
paintings = os.listdir(paintings_path)

paintings_descriptors = []

for paintFile in paintings:
    paint = cv2.imread(paintings_path + paintFile)
    if paint is not None:
        orb = cv2.ORB_create()
        paintKeypoints, trainDescriptors = orb.detectAndCompute(paint, None)
        paintings_descriptors.append({'desc':trainDescriptors, 'name':paintFile})

def allVideos():

    for videoFile in videos:

        video = cv2.VideoCapture(video_path + videoFile)
        frame_counter = 0

        while video.isOpened():

            ret, frame = video.read()
            if frame is not None:
                frame_counter += 1
                if frame_counter % 50 == 0:

                    blue, boundingBoxes = boundingBox(frame)

                    for bb in boundingBoxes:
                        cut = bb['cut']
                        segmentation(cut, paintings_descriptors)

                    cv2.namedWindow('blue', cv2.WINDOW_NORMAL)
                    cv2.moveWindow("blue", 50, 50)
                    cv2.resizeWindow("blue", 400, 300)
                    cv2.imshow('blue', blue)

            # stop
            if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                video.release()
                break

    video.release()
    cv2.destroyAllWindows()


def singleVideo():

    video = cv2.VideoCapture(path + 'GOPR1938.MP4')
    frame_counter = 0

    while video.isOpened():

        ret, frame = video.read()
        if frame is not None:
            frame_counter += 1
            if frame_counter % 20 == 0:

                blue, boundingBoxes = boundingBox(frame)

                for bb in boundingBoxes:
                    cut = bb['cut']
                    segmentation(cut)


                cv2.namedWindow('blue', cv2.WINDOW_NORMAL)
                cv2.moveWindow("blue", 50, 50)
                cv2.resizeWindow("blue", 400, 300)
                cv2.imshow('blue', blue)

        # stop
        if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            video.release()
            input("Press Enter to continue...")
            break

    video.release()
    cv2.destroyAllWindows()

allVideos()