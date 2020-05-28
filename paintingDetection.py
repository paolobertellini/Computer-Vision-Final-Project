import cv2


def paintingDetection(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.erode(frame_hsv, None, iterations=5)
    frame_hsv = cv2.medianBlur(frame_hsv, 25)

    mask = cv2.inRange(frame_hsv, (0, 0, 80), (255, 255, 255))
    mask = 255 - mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    errors = []

    for i, cnt in enumerate(contours):
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if cv2.contourArea(approx) > 15000:
            if len(approx) == 4:
                bbox = cv2.boundingRect(approx)
                x, y, w, h = bbox
                boxes.append((x, y, x + w, y + h))

    return boxes
