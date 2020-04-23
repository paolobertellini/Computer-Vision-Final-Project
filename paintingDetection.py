import numpy as np
import cv2

def paintingDetection(frame, video):

    _width = video.get(3)
    _height = video.get(4)
    _margin = 0.0
    corners = np.array(
        [
            [[_margin, _margin]],
            [[_margin, _height + _margin]],
            [[_width + _margin, _height + _margin]],
            [[_width + _margin, _margin]],
        ]
    )

    pts_dst = np.array(corners, np.float32)

    # gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 1, 10, 120)

    # edges
    Th1 = 60  # cv2.getTrackbarPos("Th1", "Trackbars")
    Th2 = 50  # cv2.getTrackbarPos("Th2", "Trackbars")
    edges = cv2.Canny(gray, Th1, Th2)

    # contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print contours
    for cont in contours:
        if cv2.contourArea(cont) > 5000:
            arc_len = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)
            if (len(approx) == 4):
                IS_FOUND = 1
                pts_src = np.array(approx, np.float32)
                h, status = cv2.findHomography(pts_src, pts_dst)
                out = cv2.warpPerspective(frame, h, (int(_width + _margin * 2), int(_height + _margin * 2)))
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            else:
                pass

    return edges, frame

