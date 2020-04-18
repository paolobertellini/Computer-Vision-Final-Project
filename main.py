import numpy as np

import cv2
def nothing(x):
    pass

cap = cv2.VideoCapture('VIRB0400.MP4')
cv2.namedWindow("Trackbars")

cv2.createTrackbar("Th1", "Trackbars", 0, 500, nothing)
cv2.createTrackbar("Th2", "Trackbars", 0, 500, nothing)


while(cap.isOpened()):
    ret, frame = cap.read()

    _width = cap.get(3)
    _height = cap.get(4)
    _margin = 0.0

    corners = np.array(
        [
            [[_margin, _margin]],
            [[_margin, _height + _margin]],
            [[_width + _margin, _height + _margin]],
            [[_width +_margin, _margin]],
        ]
    )

    pts_dst = np.array( corners,np.float32 )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 1, 10, 120)
    Th1 = 60#cv2.getTrackbarPos("Th1", "Trackbars")
    Th2 = 50#cv2.getTrackbarPos("Th2", "Trackbars")
    edges = cv2.Canny(gray, Th1, Th2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        #cv2.imshow( 'closed', closed )
        #cv2.imshow( 'gray', gray )
        cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
        cv2.imshow('edges', edges)


        cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
        cv2.imshow('rgb', frame)
        cv2.imwrite('linesDetected.jpg', frame)



    '''frame = cv2.medianBlur(frame, ksize=9)
    Z = frame.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))

    contours, hierarchy = cv2.findContours(edges,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    lines= cv2.HoughLinesP(edges, rho=1, theta=math.pi / 180, threshold=70, minLineLength=100, maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2=line[0]
        #cv2.line(frame,(x1,y1),(x2,y2), (0, 0, 255),5)
    for r, theta in lines[0]:
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # All the changes made in the input image are finally
    # written on a new image houghlines.jpg
    
    cv2.imshow('mask', res2)
    cv2.imshow('mask1', frame)'''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()