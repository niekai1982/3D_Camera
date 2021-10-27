import cv2
import os
import matplotlib as mpl
mpl.use('tkagg')
import numpy as np
from pattern_decode import decode_gray_set
import glob
import matplotlib.pyplot as plt

shadow_threshold = 10
corner_count_x = 7
corner_count_y = 11
corners_width = 21
corners_height = 21.08


chessboard_corners = []
projector_corners = []
pattern_list = []


class cvSize(object):
    def __init__(self, w=0, h=0):
        self.width = w
        self.height = h

chessboard_size = cvSize(11, 7)
corner_size = cvSize(21., 21.)

def extract_chessboard_corners(pattern_set):
    count = len(pattern_set)

    global chessboard_size
    global chessboard_corners
    global corner_size

    chessboard_size = cvSize(corner_count_x, corner_count_y)
    chessboard_corners = [[] for i in range(count)]
    corner_size = cvSize(corners_width, corners_height)

    all_found = True
    imageSize = cvSize()

    for i in range(count):
        gray_image = cv2.imread(pattern_set[i][0], 0)
        if gray_image.shape[0] < 1:
            continue
        if i == 0:
            imageSize.width = gray_image.shape[1]
            imageSize.height = gray_image.shape[0]
        elif imageSize.width != gray_image.shape[1]:
            return False
        ret, corners = cv2.findChessboardCorners(gray_image, (chessboard_size.width, chessboard_size.height), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print("find corners num is:", len(corners))
        else:
            all_found = False
        chessboard_corners[i] = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        #### visulization
        # cv2.drawChessboardCorners(gray_image, (chessboard_size.height, chessboard_size.width), chessboard_corners[i], ret)
        # plt.imshow(gray_image)
        # plt.show()

    return all_found


def calibrate(pattern_set):
    count = len(pattern_set)
    threshold = shadow_threshold
    imageSize = cvSize()

    global chessboard_corners

    if not extract_chessboard_corners(pattern_set):
        return

    world_corners = []
    for h in range(chessboard_size.height):
        for w in range(chessboard_size.width):
            world_corners.append([corner_size.width*w, corner_size.height*h, 0.])

    objectPoints = []
    for i in range(count):
        objectPoints.append(world_corners)

    global projector_corners
    projector_corners = [[] for i in range(count)]

    global pattern_list
    pattern_list = [[] for i in range(count)]

    for i in range(count):
        corners = chessboard_corners[i]
        pcorners = projector_corners[i]

        pattern_list[i], min_max_image = decode_gray_set(pattern_set[i])

        if i == 0:
            imageSize.width = pattern_list[i].shape[1]
            imageSize.height = pattern_list[i].shape[0]
        elif imageSize.width != pattern_list[i].shape[1] or imageSize.height != pattern_list[i].shape[0]:
            print("ERROR:pattern image of different size: set", i)
            return
        for  iter in corners:
            p = iter[0]
            q = []
            WINDOW_SIZE = 30
            image_points = []
            proj_points = []
            if p[0] > WINDOW_SIZE and p[1] > WINDOW_SIZE and p[0] + WINDOW_SIZE < pattern_list.shape[0] and p[1] + WINDOW_SIZE < pattern_list.shape[1]:
                for h in range(p[0] - WINDOW_SIZE, p[0] + WINDOW_SIZE):
                    for w in range(p[1] - WINDOW_SIZE, p[1] + WINDOW_SIZE):
                        pattern = pattern_list[i][w, h]
                        min_max = min_max_image[w, h]
                        if np.isnan(pattern[0]) or np.isnan(pattern[1]):
                            continue
                        if min_max[1] - min_max[0] < threshold:
                            continue
                        image_points.append([w, h])
                        proj_points.append([pattern[0], pattern[1]])
            H = cv2.findHomography(image_points, proj_points)




if __name__ == "__main__":
    pattern_file_list = glob.glob('../data/cartman/2013-May-14_20.41.56.117/*.png')
    pattern_file_list.sort()
    pattern_set = [pattern_file_list]

    extract_chessboard_corners(pattern_set)

