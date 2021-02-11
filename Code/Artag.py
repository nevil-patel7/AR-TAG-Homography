import cv2
import math
from ARtag_functions import *
import time

### Initialisation
show_lena = False
show_cube = False
lena = cv2.imread('lena.png')
tag_ids = ['0101', '0111', '1111']

## User Inputs
tag_name = input("Input tag file(without extension): ")
print("Choose what to show:\n Enter 1 for Lena\n Enter 2 for Cube")
a = int(input("Choice: "))
if a == 1:
    show_lena = True
elif a == 2:
    show_cube = True
else:
    print("sorry selection could not be identified, exiting code")
    exit(0)
video = cv2.VideoCapture(str(tag_name) + '.mp4')
video_encoder = cv2.VideoWriter_fourcc(*'XVID')
today = time.strftime("%m-%d__%H.%M.%S")
videoname = str(today) + str(tag_name) + ('lena' if show_lena == True else 'cube')
frame_rate = 20
output_video = cv2.VideoWriter(str(videoname) + ".avi", video_encoder, frame_rate, (1920, 1080))
K = np.array(
    [[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]])
start_frame = 0
count = start_frame
video.set(1, start_frame)

while video.isOpened():
    ret, frame = video.read()
    [all_cnts, cnts] = contour_detection(frame)  # Detect Contours
    cv2.drawContours(frame, all_cnts, -1, (255, 0, 0), 4)  # Draw Detected Contours
    for i, tag in enumerate(cnts):
        H = homography(tag)  # find Homography
        H_inv = np.linalg.inv(H)
        square_img = warp(H_inv, frame, 200, 200)  # Warp te image
        imgray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        ret, square_img = cv2.threshold(imgray, 180, 255, cv2.THRESH_BINARY)

        [id_str, orient] = find_id(square_img)  # Detect orientation and tagid from image

        if show_lena:
            new_img = lena
            rotated_img = reorient(new_img, orient)

            dim = rotated_img.shape[0]
            H = homography(tag,dim)
            h = frame.shape[0]
            w = frame.shape[1]
            # Superimposing 'Lena.png' on the tag with given orientation
            frame1 = warp(H, rotated_img, h, w)
            frame2 = imposelena(frame, all_cnts[i], 0)
            superimposed_frame = cv2.bitwise_or(frame1, frame2)
            cv2.imshow("Current_frame", superimposed_frame)
            output_video.write(superimposed_frame)

        # Generate and and impose an Augmented 3d Cube on the tag
        if show_cube:
            H = homography(tag, 200)
            H_inv = np.linalg.inv(H)
            P = projection_mat(K, H_inv)
            new_corners = cubePoints(tag, H, P, 200)
            frame = drawCube(tag, new_corners, frame, 0)

            cv2.imshow("edges", frame)
            output_video.write(frame)

    if cv2.waitKey(1) & 0xff == 27:
        break
video.release()
cv2.destroyAllWindows()
