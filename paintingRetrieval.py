import cv2


def paintingRetrieval(painting, paintings_info):
    orb = cv2.ORB_create()
    _, paintingDescriptors = orb.detectAndCompute(painting, None)

    paintingScore = []

    for i, pd in enumerate(paintings_info):

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(paintingDescriptors, pd['Desc'], k=2)

        score = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                score.append([m])

        ps = {'index': i, 'n': len(score), 'p': len(score) / (len(pd['Desc']) / 5) * 100}
        paintingScore.append(ps)

    paintingScore = sorted(paintingScore, key=lambda l: l['n'], reverse=True)

    info = paintings_info[paintingScore[0]['index']]['Title'][:10] + '(' + str(
        paintingScore[0]['index']) + ')' + '[' + str(paintingScore[0]['p']) + '%]'
    retrieval = cv2.imread('notfound.png')

    if paintingScore[0]['n'] >= 25:
        c = (0, 255, 0)
        retrieval = paintings_info[paintingScore[0]['index']]['Painting']

        # print(paintings_info[paintingScore[0]['index']]['Title'], '(',
        #       paintings_info[paintingScore[0]['index']]['Image'], ')')

    elif 5 < paintingScore[0]['n'] < 25:
        c = (0, 255, 255)

        # print('Similar to: ' + paintings_info[paintingScore[0]['index']]['Title'], '(',
        #       paintings_info[paintingScore[0]['index']]['Image'],
        #       ')' + '[' + str(paintingScore[0]['n']) + ']')

    else:
        c = (0, 0, 255)

    return paintingScore, c, info, retrieval
