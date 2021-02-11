import cv2
import numpy as np


# find_id takes an image as input and returns [returnstring,orient]= ID detected,Orientation information from the tag
def find_id(image):
    orient = ''
    ret, img_binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    unpadded_image = img_binary[50:150, 50:150]

    binaryTestPoints = np.array([[37, 37], [62, 37], [37, 62], [62, 62]])
    OrientTestPoints = np.array([[85, 85], [15, 85], [15, 15], [85, 15]])
    white = 255
    binarylist = []
    for i in range(0, 4):
        x = binaryTestPoints[i][0]
        y = binaryTestPoints[i][1]
        if (unpadded_image[x, y]) == white:
            binarylist.append('1')
        else:
            binarylist.append('0')

    if unpadded_image[OrientTestPoints[0][0], OrientTestPoints[0][1]] == white:
        orient = 3
    elif unpadded_image[OrientTestPoints[1][0], OrientTestPoints[1][1]] == white:
        orient = 2
    elif unpadded_image[OrientTestPoints[2][0], OrientTestPoints[2][1]] == white:
        orient = 1
    elif unpadded_image[OrientTestPoints[3][0], OrientTestPoints[3][1]] == white:
        orient = 0
    returnstring = str(binarylist)
    return returnstring, orient


# contour_detection() takes an image as an input converts it to Grayscale image finds contours sorts them and also
# differentiates between the tag and the white space,
# It returns [return_cnts,corner]= tag contours, corners
def contour_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    ret, thresh = cv2.threshold(gray, 190, 255, 0)
    all_cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    wrong_cnts = []
    for i, h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:
            wrong_cnts.append(i)
    cnts = [c for i, c in enumerate(all_cnts) if i not in wrong_cnts]

    # sort the contours to include only the three largest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    return_cnts = []
    for c in cnts:homography

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri * .015, True)
        if len(approx) == 4:
            return_cnts.append(approx)

    corners = []
    for shape in return_cnts:
        coords = []
        for p in shape:
            coords.append([p[0][0], p[0][1]])
        corners.append(coords)

    return return_cnts, corners


# Performs Homography on a fixed size sqaure image
# returns H= homography matrix (3x3)
def homography(corners, dim=200):
    # Define the eight points to compute the homography matrix
    x = []
    y = []
    for point in corners:
        x.append(point[0])
        y.append(point[1])
    # ccw corners
    xp = [0, dim, dim, 0]
    yp = [0, 0, dim, dim]

    n = 9
    m = 8
    A = np.empty([m, n])

    val = 0
    for row in range(0, m):
        if (row % 2) == 0:
            # A[row,: ] = [[-x[val]], [-y[val]], [-1], [0], [0] ,[0], [x[val]*xp[val]], [y[val]*xp[val]], [xp[val]]]
            A[row, 0] = -x[val]
            A[row, 1] = -y[val]
            A[row, 2] = -1
            A[row, 3] = 0
            A[row, 4] = 0
            A[row, 5] = 0
            A[row, 6] = x[val] * xp[val]
            A[row, 7] = y[val] * xp[val]
            A[row, 8] = xp[val]

        else:
            # A[row,: ] = [[0], [0], [0], [-x[val]], [-y[val]], [-1], [x[val] * xp[val]], [y[val] * xp[val]], [xp[val]]]
            A[row, 0] = 0
            A[row, 1] = 0
            A[row, 2] = 0
            A[row, 3] = -x[val]
            A[row, 4] = -y[val]
            A[row, 5] = -1
            A[row, 6] = x[val] * yp[val]
            A[row, 7] = y[val] * yp[val]
            A[row, 8] = yp[val]
            val += 1

    U, S, V = np.linalg.svd(A)
    x = V[-1]
    H = np.reshape(x, [3, 3])
    return H


# warp(H,src,h,w) is an alternative to the inbuilt warpPerspective() function in OpenCV
# The function significantly increases the processing time due to restricted use of remap function
def warp(H, src, h, w):
    indexy, indexx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([indexx.ravel(), indexy.ravel(), np.ones_like(indexx).ravel()])

    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1] / map_ind[-1]
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    map_x[map_x >= src.shape[1]] = -1
    map_x[map_x < 0] = -1
    map_y[map_y >= src.shape[0]] = -1
    map_x[map_y < 0] = -1

    return_img = np.zeros((h, w, 3), dtype="uint8")
    for new_x in range(w):
        for new_y in range(h):
            x = int(map_x[new_y, new_x])
            y = int(map_y[new_y, new_x])

            if x == -1 or y == -1:
                pass
            else:
                return_img[new_y, new_x] = src[y, x]

    return return_img


def reorient(image, orient):
    if orient == 1:
        reoriented_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orient == 2:
        reoriented_img = cv2.rotate(image, cv2.ROTATE_180)
    elif orient == 3:
        reoriented_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        reoriented_img = image

    return reoriented_img


def imposelena(frame, contour, color):
    cv2.drawContours(frame, [contour], -1, (color), thickness=-1)
    return frame


def projection_mat(K, H):
    h1 = H[:, 0]
    h2 = H[:, 1]

    K = np.transpose(K)

    K_inv = np.linalg.inv(K)
    a = np.dot(K_inv, h1)
    c = np.dot(K_inv, h2)
    lamda = 1 / ((np.linalg.norm(a) + np.linalg.norm(c)) / 2)

    Bhat = np.dot(K_inv, H)

    if np.linalg.det(Bhat) > 0:
        B = 1 * Bhat
    else:
        B = -1 * Bhat

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]
    r1 = lamda * b1
    r2 = lamda * b2
    r3 = np.cross(r1, r2)
    t = lamda * b3

    P = np.dot(K, (np.stack((r1, r2, r3, t), axis=1)))

    return P


def cubePoints(corners, H, P, dim):
    new_corners = []
    x = []
    y = []
    for point in corners:
        x.append(point[0])
        y.append(point[1])
    H_c = np.stack((np.array(x), np.array(y), np.ones(len(x))))

    sH_w = np.dot(H, H_c)

    H_w = sH_w / sH_w[2]

    P_w = np.stack((H_w[0], H_w[1], np.full(4, -dim), np.ones(4)), axis=0)

    sP_c = np.dot(P, P_w)
    P_c = sP_c / (sP_c[2])

    for i in range(4):
        new_corners.append([int(P_c[0][i]), int(P_c[1][i])])

    return new_corners


def drawCube(tagcorners, new_corners, frame, flag):
    thickness = 5
    if not flag:
        contours = makeContours(tagcorners, new_corners)
        for contour in contours:
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), thickness=-1)

    for i, point in enumerate(tagcorners):
        cv2.line(frame, tuple(point), tuple(new_corners[i]), (255, 0, 0), thickness)

    for i in range(4):
        if i == 3:
            cv2.line(frame, tuple(tagcorners[i]), tuple(tagcorners[0]), (255, 0, 0), thickness)
            cv2.line(frame, tuple(new_corners[i]), tuple(new_corners[0]), (255, 0, 0), thickness)
        else:
            cv2.line(frame, tuple(tagcorners[i]), tuple(tagcorners[i + 1]), (255, 0, 0), thickness)
            cv2.line(frame, tuple(new_corners[i]), tuple(new_corners[i + 1]), (255, 0, 0), thickness)

    return frame


def makeContours(corners1, corners2):
    contours = []
    for i in range(len(corners1)):
        if i == 3:
            p1 = corners1[i]
            p2 = corners1[0]
            p3 = corners2[0]
            p4 = corners2[i]
        else:
            p1 = corners1[i]
            p2 = corners1[i + 1]
            p3 = corners2[i + 1]
            p4 = corners2[i]
        contours.append(np.array([p1, p2, p3, p4], dtype=np.int32))
    contours.append(np.array([corners1[0], corners1[1], corners1[2], corners1[3]], dtype=np.int32))
    contours.append(np.array([corners2[0], corners2[1], corners2[2], corners2[3]], dtype=np.int32))

    return contours


def getCorners(frame):
    [tag_cnts, corners] = contour_detection(frame, 180)

    tag_corners = {}

    for i, tag in enumerate(corners):
        # compute homography
        dim = 200
        H = homography(tag, dim)
        H_inv = np.linalg.inv(H)

        # get squared tile
        square_img = warp(H_inv, frame, dim, dim)
        imgray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        ret, square_img = cv2.threshold(imgray, 180, 255, cv2.THRESH_BINARY)

        # encode squared tile
        [id_str, orientation] = find_id(square_img)

        ordered_corners = []

        if orientation == 0:
            ordered_corners = tag

        elif orientation == 1:
            ordered_corners.append(tag[1])
            ordered_corners.append(tag[2])
            ordered_corners.append(tag[3])
            ordered_corners.append(tag[0])

        elif orientation == 2:
            ordered_corners.append(tag[2])
            ordered_corners.append(tag[3])
            ordered_corners.append(tag[0])
            ordered_corners.append(tag[1])

        elif orientation == 3:
            ordered_corners.append(tag[3])
            ordered_corners.append(tag[0])
            ordered_corners.append(tag[1])
            ordered_corners.append(tag[2])

        tag_corners[id_str] = ordered_corners

    return tag_corners


def getTopCorners(bot_corners):
    K = np.array([[1406.08415449821, 0, 0],
                  [2.20679787308599, 1417.99930662800, 0],
                  [1014.13643417416, 566.347754321696, 1]])

    top_corners = {}

    for tag_id, corners in bot_corners.items():
        H = homography(corners, 200)
        H_inv = np.linalg.inv(H)
        P = projection_mat(K, H_inv)
        top_corners[tag_id] = cubePoints(corners, H, P, 200)

    return top_corners


def avgCorners(past, current, future):
    diff = 50
    average_corners = {}
    for tag in current:
        templist = [current[tag]]
        if past == []:
            pass
        elif tag in past[-1]:
            for d in past:
                if tag in d:
                    templist.append(d[tag])
        else:
            pass

        if tag in future[0]:
            for d in future:
                if tag in d:
                    templist.append(d[tag])
        else:
            pass

        newcorners = []
        c1x = c1y = c2x = c2y = c3x = c3y = c4x = c4y = 0

        for allcorners in templist:
            c1x += allcorners[0][0]
            c1y += allcorners[0][1]
            c2x += allcorners[1][0]
            c2y += allcorners[1][1]
            c3x += allcorners[2][0]
            c3y += allcorners[2][1]
            c4x += allcorners[3][0]
            c4y += allcorners[3][1]

        newcorners = np.array([[c1x, c1y], [c2x, c2y], [c3x, c3y], [c4x, c4y]])
        newcorners = np.divide(newcorners, len(templist))
        newcorners = newcorners.astype(int)
        newcorners = np.ndarray.tolist(newcorners)

        # If any coner value is > n pixels from original keep original
        teleport = False
        for i in range(4):
            orig_x = current[tag][i][0]
            orig_y = current[tag][i][1]
            new_x = newcorners[i][0]
            new_y = newcorners[i][1]
            if (abs(orig_x - new_x) > diff) or (abs(orig_y - new_y) > diff):
                teleport = True
        if teleport:
            average_corners[tag] = current[tag]
        else:
            average_corners[tag] = newcorners

    return average_corners
