import cv2
import os

pepp_path = '/media/peppepc/Volume/Peppe/Unimore/Vision and Cognitive Systems/Project material/'
paolo_path = 'C:/VCS-project/'
path = paolo_path
paintings_path = path + 'paintings_db/'
paintings = os.listdir(paintings_path)


image = cv2.imread("detect-1589536183.734341.jpg")
true = cv2.imread(paintings_path)

def retrieval(img, paintings_descriptors):

    ret = 'soreta'
    orb = cv2.ORB_create()
    queryKeypoints, queryDescriptors = orb.detectAndCompute(img, None)
    # trainKeypoints, trainDescriptors = orb.detectAndCompute(true, None)

    for pd in paintings_descriptors:

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(queryDescriptors, pd['desc'], k = 2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        if len(good) > 20:
            print ("similar image", pd['name'])
            ret = pd['name']

    return ret
    # final_img = cv2.drawMatches(image, queryKeypoints, true, trainKeypoints, matches[:10], None)
    # final_img = cv2.resize(final_img, (1000, 650))
    #
    # # Show the final image
    # cv2.imshow("Matches", final_img)
    # cv2.waitKey(1000000000)


