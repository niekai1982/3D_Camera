import cv2
import os
import matplotlib as mpl

mpl.use('tkagg')
import numpy as np
from reconstruct import reconstruct_model_simple
from pattern_decode import decode_gray_set
import glob
import matplotlib.pyplot as plt
from cv2python import *
import numba as nb
import open3d as o3d

# from calibrate import CalibrationData

projector_resolution = (1280, 720)
camera_resolution = (2048, 1536)

board_size = (12, 9)
board_distance_world = 20  # mm

pattern_size = (10, 8)
pattern_distance_pixel = 50  # pixels


class Board(object):
    def __init__(self, camera_resolution, board_size, board_feature_distance):
        self.pattern_img = []
        self.feature_size = cvSize(w=board_size[0] - 1, h=board_size[1] - 1)
        self.feature_distance = board_feature_distance
        self.board_size = cvSize(w=board_size[0], h=board_size[1])
        self.camera_resolution = cvSize(camera_resolution[0], camera_resolution[1])
        self.object_points_xyz = []
        self.object_points_xy = []
        self.generate_board_feature_points()

    def generate_board_feature_points(self):
        for h in range(self.feature_size.height):
            for w in range(self.feature_size.width):
                self.object_points_xyz.append([self.feature_distance * w, self.feature_distance * h, 0.])
                self.object_points_xy.append([self.feature_distance * w, self.feature_distance * h])
        self.object_points_xy = np.array(self.object_points_xy)
        self.object_points_xy.shape = -1, 2
        self.object_points_xyz = np.array(self.object_points_xyz)
        self.object_points_xyz.shape = -1, 3


class Pattern(object):
    def __init__(self, projector_resolution, pattern_size, pattern_feature_distance):
        self.pattern_img = []
        self.feature_size = cvSize(w=pattern_size[0] - 1, h=pattern_size[1] - 1)
        self.feature_distance = pattern_feature_distance
        self.pattern_size = cvSize(w=pattern_size[0], h=pattern_size[1])
        self.projector_resolution = cvSize(projector_resolution[0], projector_resolution[1])
        self.feature_location_in_projector = []
        self.generate_pattern_feature_location_in_projector()

    def generate_pattern_feature_location_in_projector(self):
        # x, y
        spacing = self.feature_distance
        square_size = self.feature_distance
        width = self.projector_resolution.width
        height = self.projector_resolution.height
        cols = self.pattern_size.width
        rows = self.pattern_size.height
        xspacing = (width - cols * square_size) // 2
        yspacing = (height - rows * square_size) // 2
        for y in range(1, rows):
            for x in range(1, cols):
                self.feature_location_in_projector.append([x * spacing + xspacing, y * spacing + yspacing])
        self.feature_location_in_projector = np.array(self.feature_location_in_projector)
        self.feature_location_in_projector.shape = -1, 2


class CalibrationData(object):
    def __init__(self, camera_resolution, projector_resolution):
        self.cam_K = np.empty(shape=(3, 3), dtype=np.float32)
        self.cam_kc = np.empty(shape=(1, 5), dtype=np.float32)
        self.proj_K = np.empty(shape=(3, 3), dtype=np.float32)
        self.proj_kc = np.empty(shape=(1, 5), dtype=np.float32)
        self.R = np.empty(shape=(3, 3), dtype=np.float32)
        self.T = np.empty(shape=(1, 3), dtype=np.float32)
        self.cam_error = 0.
        self.proj_error = 0.
        self.stereo_error = 0.
        self.filename = ''
        self.camera_is_valid = False
        self.projector_is_valid = False
        self.camera_resolution = cvSize(w=camera_resolution[0], h=camera_resolution[1])
        self.projector_resolution = cvSize(w=projector_resolution[0], h=projector_resolution[1])
        self.pose_num = 0
        self.homography = []

    def load_calibration(self, filename):
        pass

    def save_calibration(self, filename):
        pass

    def load_calibration_yaml(self, filename):
        pass

    def save_calibration_yaml(self, filename):
        pass


def extract_chessboard_corners(pattern_image, chessboard_size, chessbord_flag=True, invert=False, vis_flag=False):
    # count = len(pattern_set)
    # all_found = True
    scale = 4
    chessboard_cols = chessboard_size.width
    chessboard_rows = chessboard_size.height
    # chessboard_corners = []

    # for i in range(count):
    # gray_image = cv2.imread(pattern_set[i], 0)
    gray_image = pattern_image.copy()
    if invert:
        gray_image = 255 - gray_image
    w, h = gray_image.shape[1], gray_image.shape[0]
    if chessbord_flag:
        scaled_image = cv2.resize(gray_image, (w // scale, h // scale))
        ret, corners = cv2.findChessboardCorners(scaled_image, (chessboard_cols, chessboard_rows))
        if ret:
            corners = corners * scale
            cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    else:
        try:
            ret, corners = cv2.findCirclesGrid(gray_image, (7, 7), cv2.CALIB_CB_SYMMETRIC_GRID)
            print(ret)
        except cv2.error as e:
            print(e)

    if ret:
        print("find corners num is:", len(corners))
        # chessboard_corners.append(corners.reshape(-1, 2))
    if vis_flag:
        vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        cv2.drawChessboardCorners(vis, (chessboard_rows, chessboard_cols), corners, ret)
        plt.imshow(vis)
        plt.show()
    return ret, corners


def calibrate_camera(boards_features, boards_points_xy, boards_points_xyz, calib_data):
    pose_num = len(boards_features)

    criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 50)
    # cal_flag = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
    cal_flag = 0 + cv2.CALIB_FIX_K3
    cal_flag += cv2.CALIB_ZERO_TANGENT_DIST

    objectPoints = boards_points_xyz
    # objectPoints.shape = -1, 3

    imagePoints = boards_features
    # imagePoints.shape = -1, 2

    cam_ret, cam_k, cam_kc, cam_rvecs, cam_tvecs = cv2.calibrateCamera(objectPoints, imagePoints, (
    calib_data.camera_resolution.width, calib_data.camera_resolution.height), cal_flag, criteria)
    homography_list = []

    for i in range(pose_num):
        board_feature = boards_features[i]
        board_feature_undistorted = cv2.undistortPoints(board_feature, cam_k, cam_kc)
        homography = cv2.findHomography(board_feature_undistorted, boards_points_xy[i])
        homography_list.append(homography)
    calib_data.cam_error = cam_ret
    calib_data.cam_K = cam_k
    calib_data.cam_kc = cam_kc
    calib_data.homography = homography_list
    calib_data.camera_is_valid = True


# @nb.jit(nopython=True)
def RemovePrinted_AddProjectBoard(projector_all_on, projector_all_off, board_image_print_and_project):
    albedo = np.empty_like(projector_all_on)
    out = np.empty_like(projector_all_on)
    all_on = projector_all_on.copy()
    all_off = projector_all_off.copy()
    combo = board_image_print_and_project.copy()

    rows, cols = projector_all_on.shape
    # for yRow in range(rows):
    #     for xCol in range(cols):
    #         val_all_off = all_off[yRow, xCol]
    #         val_all_on = all_on[yRow, xCol]
    #         val_albedo = (val_all_on + val_all_off) // 2
    #         albedo[yRow, xCol] = val_albedo
    # for yRow in range(rows):
    #     for xCol in range(cols):
    #         val_coded = combo[yRow, xCol]
    #         val_albedo = albedo[yRow, xCol]
    #         if val_coded > val_albedo or abs(val_coded-val_albedo) < 5:
    #             out[yRow, xCol] = 255
    #         else:
    #             out[yRow, xCol] = 0
    # return out
    albedo = (all_on + all_off) / 2
    out = np.zeros_like(out)
    out[combo > albedo] = 255
    # out[np.abs(combo - albedo) < 1] = 255
    return out


def calibrate_projector(patterns_features, pattern_features_in_projector, calib_data):
    pose_num = len(patterns_features)
    patterns_points_xyz = []
    patterns_points_xy = []
    for i in range(pose_num):
        pattern_features = cv2.undistortPoints(patterns_features[i], calib_data.cam_K, calib_data.cam_kc)
        pattern_points_xy = cv2.perspectiveTransform(pattern_features, calib_data.homography[i][0])
        patterns_points_xy.append(pattern_points_xy)
        patterns_points_xyz.append(np.dstack((pattern_points_xy, np.zeros_like(pattern_points_xy[:, :, :1]))))
    criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 50)
    # cal_flag = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
    cal_flag = 0 + cv2.CALIB_FIX_K3
    cal_flag += cv2.CALIB_ZERO_TANGENT_DIST
    projector_ret, projector_k, projector_kc, projector_rvecs, projector_tvecs = cv2.calibrateCamera(patterns_points_xyz, pattern_features_in_projector, (calib_data.projector_resolution.width, calib_data.projector_resolution.height), cal_flag, criteria)
    calib_data.proj_error = projector_ret
    calib_data.proj_K = projector_k
    calib_data.proj_kc = projector_kc
    calib_data.projector_is_valid = True
    return patterns_points_xyz


def calibrate(pattern_set):
    count = len(pattern_set)
    pose_num = count // 4

    board = Board(camera_resolution, board_size, board_distance_world)
    pattern = Pattern(projector_resolution, pattern_size, pattern_distance_pixel)

    calib_data = CalibrationData(camera_resolution, projector_resolution)

    if pose_num * 4 != count:
        print("pattern number error!")
        return False

    boards_features = []
    boards_points_xy = []
    boards_points_xyz = []

    patterns_features = []
    patterns_features_in_projector = []
    patterns_points_xy = []
    patterns_points_xyz = []

    for i in range(pose_num):
        projector_all_on = cv2.imread(pattern_set[i * 4], 0)
        projector_all_off = cv2.imread(pattern_set[i * 4 + 1], 0)
        board_and_pattern_image = cv2.imread(pattern_set[i * 4 + 2], 0)

        # STEP 1 remove board
        pattern_image = RemovePrinted_AddProjectBoard(projector_all_on, projector_all_off, board_and_pattern_image)

        # STEP 2 extract feature location
        ret_b, board_feature_location = extract_chessboard_corners(projector_all_on, board.feature_size)
        ret_p, pattern_feature_location = extract_chessboard_corners(pattern_image, pattern.feature_size)

        if ret_b and ret_b:
            boards_features.append(np.float32(board_feature_location.reshape(-1, 2)))
            boards_points_xy.append(np.float32(board.object_points_xy))
            boards_points_xyz.append(np.float32(board.object_points_xyz))

            patterns_features.append(np.float32(pattern_feature_location.reshape(-1, 2)))
            patterns_features_in_projector.append(np.float32(pattern.feature_location_in_projector))

    calibrate_camera(boards_features, boards_points_xy, boards_points_xyz, calib_data)
    patterns_points_xyz = calibrate_projector(patterns_features, patterns_features_in_projector, calib_data)


    cal_flag = 0 + cv2.CALIB_FIX_K3
    cal_flag += cv2.CALIB_ZERO_TANGENT_DIST
    _, _, _, _, _, R, T, E, F =  cv2.stereoCalibrate(patterns_points_xyz, patterns_features, patterns_features_in_projector, calib_data.cam_K, calib_data.cam_kc, calib_data.proj_K, calib_data.proj_kc, (calib_data.camera_resolution.width, calib_data.camera_resolution.height),
                                                     (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS,150), cv2.CALIB_FIX_INTRINSIC + cal_flag)

    calib_data.R = R
    calib_data.T = T
    return calib_data

if __name__ == "__main__":
    pattern_set = glob.glob('../data/TI/cd_11_6_1/*.jpg')
    pattern_set.sort()
    calib_data = calibrate(pattern_set)

    data_path = "../data/TI/savedimages_bearing2"
    # pattern_path_set = [os.path.join(data_path, elem) for elem in os.listdir(data_path) if not elem.startswith('.')]
    # pattern_path_set.sort()

    pattern_set = []
    # for pattern_folder in pattern_path_set:
    #     print(pattern_folder)
    pattern_file_list = glob.glob(data_path + "/*.jpg")
    pattern_file_list.sort()
    count = len(pattern_file_list)
    bit_number = 10
    pattern_file_calibration_list = []
    pattern_file_calibration_list.append(pattern_file_list[-2])
    pattern_file_calibration_list.append(pattern_file_list[-1])
    for idx in range(2*bit_number):
        pattern_file_calibration_list.append(pattern_file_list[idx])
    for idx in range(2*bit_number):
        pattern_file_calibration_list.append(pattern_file_list[2*bit_number+2+2 + idx])
    pattern_set.append(pattern_file_calibration_list)
    # calib = CalibrationData()
    # calib.cam_K = out[0][0]
    # calib.cam_kc = out[0][1]
    # calib.proj_K = out[1][0]
    # calib.proj_kc = out[1][1]
    # calib.R = out[2][0]
    # calib.T = out[2][1]
    # calib.is_valid = True

    out = reconstruct_model_simple(calib_data, pattern_set[0])
    np.save('./test_bearing_3.npy', out)

    # out[out[:,:,2]>1000] = 0
    # out[out[:,:,2]<100] = 0
    # out[out<1000] = 0
    xyz = np.zeros((out.shape[0] * out.shape[1], 3), dtype=np.float32)

    out[out[:,:,2]>1000] = 0
    out[out[:,:,2]<100] = 0

    xyz[:, 0] = out[:, :, 1].flatten()
    xyz[:, 1] = out[:, :, 0].flatten()
    xyz[:, 2] = out[:, :, 2].flatten()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("./test_bearing_3.ply", pcd)

    # print(out)
    # print(pattern_set)
