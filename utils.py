import csv

import cv2

p = 60
b = p - 5


def loadPaintingsInfo(root_path):
    paintingsDB_path = root_path + 'paintings_db/'
    paintings_info_file = root_path + 'data.csv'
    with open(paintings_info_file) as file:
        paintings_info = list(csv.DictReader(file, delimiter=','))

    print("Loading db..")
    for paintFile in paintings_info:
        paint = cv2.imread(paintingsDB_path + paintFile['Image'])
        if paint is not None:
            orb = cv2.ORB_create()
            paintKeypoints, paintDescriptors = orb.detectAndCompute(paint, None)
            paintFile['Desc'] = paintDescriptors
            paintFile['Painting'] = paint

    return paintings_info


def cut(frame, bbox):
    bordered = cv2.copyMakeBorder(frame, p, p, p, p, borderType=cv2.BORDER_CONSTANT)
    x, y, w, h = bbox
    return bordered[y:y + h + 2 * b, x:x + w + 2 * b]


def removeInnerBox(peopleBoxes, paintingsBoxes):
    boxes = []
    errors = []
    [boxes.append(i) for i in peopleBoxes]
    [boxes.append(i) for i in paintingsBoxes]

    if boxes is not None:
        for x11, y11, x12, y12 in boxes:
            for x21, y21, x22, y22 in boxes:
                if ((x11 > x21) & (y11 > y21) & (x12 < x22) & (y12 < y22)):
                    errors.append((x11, y11, x12, y12))
        if len(errors) > 0:
            peopleBoxes = [b for b in peopleBoxes if b not in errors]
            paintingsBoxes = [b for b in paintingsBoxes if b not in errors]

        return peopleBoxes, paintingsBoxes


def resize(dim, img):
    r = dim / img.shape[1]
    dim = (dim, int(img.shape[0] * r))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
