import cv2
import os
import matplotlib as mpl
# mpl.use('tkagg')
import numpy as np
from pattern_decode import decode_gray_set
import glob
import matplotlib.pyplot as plt
from cv2python import *

shadow_threshold = 10
corner_count_x = 8
corner_count_y = 8
corners_width = 15.
corners_height = 15.


chessboard_corners = []
projector_corners = []
pattern_list = []



chessboard_size = cvSize(8, 8)
corner_size = cvSize(15., 15.)
projector_size = cvSize(1600, 1200)

class CalibrationData(object):
    def __init__(self):
        self.cam_K = np.empty(shape=(3,3), dtype=np.float32)
        self.cam_kc = np.empty(shape = (1,5), dtype=np.float32)
        self.proj_K = np.empty(shape=(3,3), dtype=np.float32)
        self.proj_kc = np.empty(shape=(1, 5), dtype=np.float32)
        self.R =np.empty(shape=(3,3), dtype=np.float32)
        self.T = np.empty(shape=(1,3), dtype=np.float32)
        self.cam_error = 0.
        self.proj_error = 0.
        self.stereo_error = 0.
        self.filename = ''
        self.is_valid = False

    def load_calibration(self, filename):
        pass
    def save_calibration(self, filename):
        pass
    def load_calibration_yaml(self, filename):
        pass
    def save_calibration_yaml(self, filename):
        pass


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
        # ret, corners = cv2.findChessboardCorners(gray_image, (chessboard_size.width, chessboard_size.height), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv2.findChessboardCorners(gray_image, (chessboard_size.width, chessboard_size.height))
        plt.imshow(gray_image)
        plt.show()
        if ret:
            print("find corners num is:", len(corners))
        else:
            all_found = False
        cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        chessboard_corners[i] = corners.reshape(-1, 2)

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
        world_corners = np.array(world_corners, dtype=np.float32)
        world_corners.shape = -1, 3
        objectPoints.append(world_corners)

    global projector_corners
    projector_corners = [[] for i in range(count)]

    global pattern_list
    pattern_list = [[] for i in range(count)]


    for i in range(count):
        corners = chessboard_corners[i]
        pcorners = projector_corners[i]

        # pattern_image, min_max_image = decode_gray_set(pattern_set[i])
        if i > 2:
            pattern_image = np.load('./Nikon_' + str(i+2)+'_pattern_image.npy')
            min_max_image = np.load('./Nikon_' + str(i+2)+'_min_max_image.npy')
        else:
            pattern_image = np.load('./Nikon_' + str(i+1)+'_pattern_image.npy')
            min_max_image = np.load('./Nikon_' + str(i+1)+'_min_max_image.npy')

        if i == 0:
            imageSize.width = pattern_image.shape[1]
            imageSize.height = pattern_image.shape[0]
        elif imageSize.width != pattern_image.shape[1] or imageSize.height != pattern_image.shape[0]:
            print("ERROR:pattern image of different size: set", i)
            return
        for  p in corners:
            # p = iter
            WINDOW_SIZE = 30
            image_points = []
            proj_points = []
            if p[0] > WINDOW_SIZE and p[1] > WINDOW_SIZE and p[0] + WINDOW_SIZE < pattern_image.shape[1] and p[1] + WINDOW_SIZE < pattern_image.shape[0]:
                for h in range(int(p[1]) - WINDOW_SIZE, int(p[1]) + WINDOW_SIZE):
                    for w in range(int(p[0]) - WINDOW_SIZE, int(p[0]) + WINDOW_SIZE):
                        pattern = pattern_image[h, w]
                        min_max = min_max_image[h, w]
                        if np.isnan(pattern[0]) or np.isnan(pattern[1]):
                            continue
                        if min_max[1] - min_max[0] < threshold:
                            continue
                        image_points.append([w, h])
                        proj_points.append([pattern[0], pattern[1]])
                H = cv2.findHomography(np.array(image_points), np.array(proj_points), cv2.RANSAC)
                q_w = np.matmul(H[0], np.array([p[0], p[1],1.]))
                q = [q_w[0]/q_w[2], q_w[1]/q_w[2]]
            else:
                return
            pcorners.append(q)
        projector_corners[i] = np.array(pcorners, dtype=np.float32)

    ### calibration the camera
    criteria = (cv2.TermCriteria_COUNT+cv2.TermCriteria_EPS, 50)
    # cal_flag = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
    cal_flag = 0 + cv2.CALIB_FIX_K3
    car_ret, cam_k, cam_kc, cam_rvecs, cam_tvecs = cv2.calibrateCamera(objectPoints, chessboard_corners, (imageSize.width, imageSize.height), cal_flag, criteria)

    ### calibrate the projector
    proj_ret, proj_k, proj_kc, proj_rvecs, proj_tvecs = cv2.calibrateCamera(objectPoints, projector_corners, (projector_size.width, projector_size.height), cal_flag, criteria)

    ### stereo calibration
    _, _, _, _, _, R, T, E, F =  cv2.stereoCalibrate(objectPoints, chessboard_corners, projector_corners, cam_k, cam_kc, proj_k, proj_kc, (imageSize.width, imageSize.height),
    (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS,150), cv2.CALIB_FIX_INTRINSIC + cal_flag)

    return (cam_k, cam_kc, cam_rvecs, cam_tvecs), (proj_k, proj_kc, proj_rvecs, proj_tvecs), (R, T, E, F)

if __name__ == "__main__":

    pattern_set = []
    for i in [1,2,3,5,6,7,8]:
        pattern_file_list = glob.glob('../data/Nikon/' + str(i) + '/*.JPG')
        pattern_file_list.sort()
        pattern_set.append(pattern_file_list)

    out = calibrate(pattern_set)
    print(out)

