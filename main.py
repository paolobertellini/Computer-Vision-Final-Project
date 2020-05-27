import csv
import os

import cv2
import numpy as np
from paintingDetection import paintingDetection
from paintingRetrieval import paintingRetrieval
from perspectiveRectification import perspectiveRectification
from peopleDetection import peopleDetection
from paintingSegmentation import paintingSegmentation

p = 60
b = p - 5
save = False


def cut(bordered, bbox):
    x, y, w, h = bbox
    return bordered[y:y + h + 2 * b, x:x + w + 2 * b]

def removeInnerBox(people, paintings):
    boxes=[]

    [boxes.append(i) for i in people]
    [boxes.append(i) for i in paintings]
    errors = []
    if boxes is not None:
        for x11, y11, x12, y12 in boxes:
            for x21, y21, x22, y22 in boxes:
                if ((x11 > x21) & (y11 > y21) & (x12 < x22) & (y12 < y22)):
                    errors.append((x11, y11, x12, y12))
        if len(errors) > 0:
            people = [b for b in people if b not in errors]
            paintings = [b for b in paintings if b not in errors]

        return people, paintings



def resize(dim, img):
    r = dim / rect_painting.shape[1]
    dim = (dim, int(rect_painting.shape[0] * r))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

best = ['GOPR1924.MP4','GOPR1940.MP4','20180206_114604.mp4','IMG_9620.MOV','IMG_4076.MOV', 'VIRB0413.MP4', 'IMG_3819.MOV', 'GOPR2039.MP4', 'IMG_2657.MOV',
        'IMG_2657.MOV','VIRB0426.MP4','GOPR1927.MP4', 'IMG_4075.MOV', 'IMG_4081.MOV', 'IMG_4086.MOV', '20180206_112658.mp4']

paolo_path = 'D:/VCS-project/'
dav_path = '/media/davide/aukey/progetto_vision/'
pepp_path = '/media/peppepc/Volume/Peppe/Unimore/Vision and Cognitive Systems/Project material/'

root_path = dav_path

paintingsDB_path = root_path + 'paintings_db/'
videos_path = root_path + 'videos/'

for root, dirs, files in os.walk(videos_path, topdown=False):
    for name in files:
        pass

videos = os.listdir(videos_path)
videos = np.random.permutation(videos)
paintings = os.listdir(paintingsDB_path)
paintings_descriptors = []

paintings_info_file = root_path + 'data.csv'
with open(paintings_info_file) as file:
    paintings_info = list(csv.DictReader(file, delimiter=','))

for paintFile in paintings_info:
    paint = cv2.imread(paintingsDB_path + paintFile['Image'])
    if paint is not None:
        orb = cv2.ORB_create()
        paintKeypoints, paintDescriptors = orb.detectAndCompute(paint, None)
        paintFile['Desc'] = paintDescriptors
        paintFile['Painting'] = paint

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.moveWindow("Detection", 10, 20)
cv2.resizeWindow("Detection", 500, 400)

cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
cv2.moveWindow("Segmentation", 10, 480)
cv2.resizeWindow("Segmentation", 300, 200)

cv2.namedWindow('Rectification', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Rectification", 520, 20)

cv2.namedWindow('Retrieval', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Retrieval", 950, 20)

for videoFile in videos:
    video = cv2.VideoCapture(videos_path + videoFile)
    frame_counter = 0
    print(videoFile)
    while video.isOpened():
        _, frame = video.read()
        if frame is not None:
            framePeople = frame.copy()
            frame_counter += 1
            if frame_counter % 4 == 0:
                box,classe=peopleDetection(frame, threshold=0.8)

                bbpaintings = paintingDetection(frame)
                if ((box is not None)& (bbpaintings is not None)):
                    box, bbpaintings = removeInnerBox(box, bbpaintings)

                frameSegmented = frame.copy()
                red_frame = np.full(frame.shape, (0, 0, 255), np.uint8)
                frameSegmented = cv2.addWeighted(frameSegmented, 0.4, red_frame, 0.6, 0)
                detections = []
                bordered = cv2.copyMakeBorder(frame, p, p, p, p, borderType=cv2.BORDER_CONSTANT)
                for x1, y1, x2, y2 in bbpaintings:
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    cutted = cut(bordered, bbox)
                    paintingSegmented = paintingSegmentation(cutted)
                    det = {'segmentation': paintingSegmented, 'bbox': bbox, 'cutted': cutted}
                    detections.append(det)

                for pd in detections:
                    x, y, w, h = pd['bbox']
                    c=(0,0,255)
                    if pd['segmentation'] is not None:

                        cv2.drawContours(frameSegmented,
                                         [pd['segmentation'] + [pd['bbox'][0] - 60, pd['bbox'][1] - 60]], 0,
                                         (0, 255, 0), -1)

                        rect_painting = perspectiveRectification(pd['cutted'], pd['segmentation'])

                        if np.size(rect_painting) != 0:
                            cv2.imshow("Rectification", resize(400, rect_painting))

                        paintingScore, c, info, retrieval = paintingRetrieval(pd['cutted'], paintings_info)


                        cv2.putText(frame, info, (x, y - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

                        cv2.imshow("Retrieval", resize(400, retrieval))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), c, 5)
                if box is not None:
                    for bp in box:
                        x1, y1, x2, y2 = bp
                        cv2.rectangle(framePeople, (x1, y1), (x2, y2 ), c, 5)

                cv2.imshow('Detection', frame)
                cv2.imshow('Segmentation', frameSegmented)
                cv2.namedWindow('giuseppe vai a cacare', cv2.WINDOW_NORMAL)
                cv2.imshow('giuseppe vai a cacare', framePeople)
            # stop
            if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(
                    cv2.CAP_PROP_FRAME_COUNT) - 1:
                video.release()
                break

video.release()
cv2.destroyAllWindows()
