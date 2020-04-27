import cv2
import numpy as np

# .   @param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if
# .   dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has
# .   half as big width and height.

# .   @param minDist Minimum distance between the centers of the detected circles. If the parameter is
# .   too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is
# .   too large, some circles may be missed.

# .   @param param1 First method-specific parameter. In case of #HOUGH_GRADIENT , it is the higher
# .   threshold of the two passed to the Canny edge detector (the lower one is twice smaller).

# .   @param param2 Second method-specific parameter. In case of #HOUGH_GRADIENT , it is the
# .   accumulator threshold for the circle centers at the detection stage. The smaller it is, the more
# .   false circles may be detected. Circles, corresponding to the larger accumulator values, will be
# .   returned first.

def circleDetection(frame):

    k1 = cv2.getTrackbarPos("k1", "Circles")
    dist_centri = cv2.getTrackbarPos("dist_centri", "Circles")
    p1 = cv2.getTrackbarPos("p1", "Circles")
    false_circle = cv2.getTrackbarPos("false_circle", "Circles")
    minR = cv2.getTrackbarPos("minR", "Circles")
    maxR = cv2.getTrackbarPos("maxR", "Circles")
    th1 = cv2.getTrackbarPos("th1", "Circles")
    th2 = cv2.getTrackbarPos("th2", "Circles")

    edges = cv2.Canny(frame, th1, th2)

    detected_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, k1, dist_centri,
                                        param1=p1, param2=false_circle, minRadius=minR, maxRadius=maxR)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for circle in detected_circles[0, :]:
            a, b, r = circle[0], circle[1], circle[2]
            cv2.circle(frame, (a, b), r, (255, 0, 0), 2)

    return edges, frame