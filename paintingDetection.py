import cv2

def paintingDetection(frame):

    blur = cv2.medianBlur(frame,21)

    frame1 = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #frame2= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    Hl = cv2.getTrackbarPos("Hl", "Trackbars")
    Sl = cv2.getTrackbarPos("Sl", "Trackbars")
    Vl = cv2.getTrackbarPos("Vl", "Trackbars")
    Hg = cv2.getTrackbarPos("Hg", "Trackbars")
    Sg = cv2.getTrackbarPos("Sg", "Trackbars")
    Vg = cv2.getTrackbarPos("Vg", "Trackbars")

    mask = cv2.inRange(frame1, (Hl,Sl,Vl), (Hg, Sg, Vg))

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) ==4:
            if cv2.contourArea(approx) > 100000:
                cv2.drawContours(blur, [approx], 0, (0, 255, 0), 3)
    cv2.drawContours(blur, approx, -1, (0, 255, 0), 3)


   # detected_circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=50,
   #                                         param2=30, minRadius=500, maxRadius=900)
   #
   # if detected_circles is not None:
   #     detected_circles = np.uint16(np.around(detected_circles))
   #     for pt in detected_circles[0, :]:
   #         a, b, r = pt[0], pt[1], pt[2]
   #          Draw the circumference of the circle.
   #         cv2.circle(frame, (a, b), r, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', mask)
    return  blur