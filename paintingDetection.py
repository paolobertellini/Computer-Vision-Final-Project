import cv2

from paintingSegmentation import paintingSegmentation

p = 60
b = p - 5
save = False


def cut(bordered, bbox):
    x, y, w, h = bbox
    return bordered[y:y + h + 2 * b, x:x + w + 2 * b]


def paintingDetection(frame):
    detections = []
    bordered = cv2.copyMakeBorder(frame, p, p, p, p, borderType=cv2.BORDER_CONSTANT)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.erode(frame_hsv, None, iterations=13)
    frame_hsv = cv2.medianBlur(frame_hsv, 31)

    mask = cv2.inRange(frame_hsv, (0, 20, 80), (255, 255, 255))
    mask = 255 - mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    errors = []

    for i, cnt in enumerate(contours):
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if cv2.contourArea(approx) > 20000:
            if len(approx) == 4:
                bbox = cv2.boundingRect(approx)
                x, y, w, h = bbox
                boxes.append((x, y, x + w, y + h))

    for x11, y11, x12, y12 in boxes:
        for x21, y21, x22, y22 in boxes:
            if ((x11 > x21) & (y11 > y21) & (x12 < x22) & (y12 < y22)):
                errors.append((x11, y11, x12, y12))
                print('DELETED BOX: ', errors)

    if len(errors) > 0:
        boxes = [b for b in boxes if b not in errors]

    for x1, y1, x2, y2 in boxes:
        bbox = [x1, y1, x2 - x1, y2 - y1]
        cutted = cut(bordered, bbox)
        paintingSegmented = paintingSegmentation(cutted)
        det = {'segmentation': paintingSegmented, 'bbox': bbox, 'cutted': cutted}
        detections.append(det)

    return detections
