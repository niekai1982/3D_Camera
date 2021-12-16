import cv2
import time
import numpy as np
import os
import glob
import matplotlib as mpl
# from calibrate import CalibrationData
import open3d as o3d
import numba as nb

# mpl.use("tkagg")
import matplotlib.pyplot as plt
from pattern_decode import decode_gray_set
from cv2python import *

THRESHOLD_DEFAULT = 12
MAX_DIST_DEFAULT = 100.
projector_size = cvSize(1280, 720)


# projector_size = cvSize(1600, 1200)

# calib = CalibrationData()
# calib.cam_K = np.array([[4.55870035e+03, 0.00000000e+00, 1.51017862e+03],
#                         [0.00000000e+00, 4.55911080e+03, 9.88716486e+02],
#                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
#
# calib.cam_kc = np.array([[-1.95063253e-01],
#                          [1.93858801e+00],
#                          [-4.99197011e-04],
#                          [1.35514749e-03],
#                          [-1.55614418e+01]], dtype=np.float32)
#
# calib.proj_K = np.array([[3.90467278e+03, 0.00000000e+00, 8.02215392e+02],
#                          [0.00000000e+00, 3.90997537e+03, 6.01700509e+02],
#                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
#
# calib.proj_kc = np.array([[-4.78490045e-01],
#                           [3.86902030e+01],
#                           [-8.60596372e-02],
#                           [1.31578421e-02],
#                           [-6.33953261e+02]], dtype=np.float32)
#
# calib.R = np.array([[0.93488111, -0.06929929, 0.34813061],
#                     [-0.02243265, 0.9672622, 0.25278571],
#                     [-0.35425145, -0.24413408, 0.90272059]], dtype=np.float32)
#
# calib.T = np.array([[-319.41056464],
#                     [-230.99720084],
#                     [525.08697085]], dtype=np.float32)
# calib.is_valid = True


def triangulate_stereo(K1, kc1, K2, kc2, Rt, T, p1, p2, distance):
    inp1 = np.empty(shape=(1, 1, 2), dtype=np.float64)
    inp2 = np.empty(shape=(1, 1, 2), dtype=np.float64)
    inp1[0, 0, 0] = p1[0]
    inp1[0, 0, 1] = p1[1]
    inp2[0, 0, 0] = p2[0]
    inp2[0, 0, 1] = p2[1]
    outp1 = cv2.undistortPoints(inp1, K1, kc1)
    outp2 = cv2.undistortPoints(inp2, K2, kc2)

    outvec1 = outp1[0, 0]
    outvec2 = outp2[0, 0]
    u1 = np.array([outvec1[0], outvec1[1], 1.0])
    u2 = np.array([outvec2[0], outvec2[1], 1.0])
    u1.shape = -1, 1
    u2.shape = -1, 1
    w1 = u1
    w2 = np.matmul(Rt, (u2 - T))

    v1 = w1
    v2 = np.matmul(Rt, u2)

    p3d, distance, _, _ = approximate_ray_intersection(v1, w1, v2, w2, distance)
    return p3d, distance


def approximate_ray_intersection(v1, q1, v2, q2, distance, out_lambda1=0, out_lambda2=0):
    v1mat = np.array(v1)
    v2mat = np.array(v2)
    v1tv1 = np.matmul(v1mat.T, v1mat)
    v2tv2 = np.matmul(v2mat.T, v2mat)
    v1tv2 = np.matmul(v1mat.T, v2mat)
    v2tv1 = np.matmul(v2mat.T, v1mat)

    Vinv = np.empty(shape=(2, 2), dtype=np.float64)
    detV = v1tv1 * v2tv2 - v1tv2 * v2tv1
    Vinv[0, 0] = v2tv2 / detV
    Vinv[0, 1] = v1tv2 / detV
    Vinv[1, 0] = v2tv1 / detV
    Vinv[1, 1] = v1tv1 / detV

    q2_q1 = q2 - q1

    Q1 = v1[0] * q2_q1[0] + v1[1] * q2_q1[1] + v1[2] * q2_q1[2]
    Q2 = -(v2[0] * q2_q1[0] + v2[1] * q2_q1[1] + v2[2] * q2_q1[2])

    lambda1 = (v2tv2 * Q1 + v1tv2 * Q2) / detV
    lambda2 = (v2tv1 * Q1 + v1tv1 * Q2) / detV

    p1 = lambda1 * v1 + q1
    p2 = lambda2 * v2 + q2

    p = 0.5 * (p1 + p2)

    distance = cv2.norm(p2 - p1)
    if out_lambda1:
        out_lambda1 = lambda1
    if out_lambda2:
        out_lambda2 = lambda2
    return p, distance, out_lambda1, out_lambda2


def reconstruct_point(pattern_image, min_max_image, cam_K, cam_kc, proj_K, proj_kc, R, T):
    threshold = THRESHOLD_DEFAULT
    max_dist = MAX_DIST_DEFAULT

    plane_dist = 100.0
    out_cols = int(pattern_image.shape[1])
    out_rows = int(pattern_image.shape[0])
    pointcloud = np.zeros(shape=(out_rows, out_cols, 3), dtype=np.float32)

    Rt = R.T

    good = 0
    bad = 0
    invalid = 0
    repeated = 0

    for h in range(pattern_image.shape[0]):
        for w in range(pattern_image.shape[1]):
            distance = max_dist
            pattern = pattern_image[h, w]
            min_max = min_max_image[h, w]
            if np.isnan(pattern[0]) or np.isnan(pattern[1]) or pattern[0] < 0. or pattern[1] < 0. or (
                    min_max[1] - min_max[0]) < threshold:
                invalid += 1
                continue
            col = pattern[0]
            row = pattern[1]

            # if projector_size.width <= int(col) or projector_size.height <= int(row):
            if 1280 <= int(col) or 720 <= int(row):
                continue

            p1 = (w, h)
            p2 = (col, row)

            p, distance = triangulate_stereo(cam_K, cam_kc, proj_K, proj_kc, Rt, T, p1, p2, distance)

            if distance < 2 * max_dist:
                d = plane_dist + 1
                if d > plane_dist:
                    pointcloud[h, w][0] = p[0][0]
                    pointcloud[h, w][1] = p[1][0]
                    pointcloud[h, w][2] = p[2][0]
    return pointcloud


def reconstruct_model_simple(calib, pattern_list):
    pattern_image, min_max_image = decode_gray_set(pattern_list)
    # pattern_image = np.load('./Nikon_test1_pattern_image.npy')
    # min_max_image = np.load('./Nikon_test1_min_max_image.npy')
    color_image = cv2.imread(pattern_list[0])
    pointcloud = reconstruct_point(pattern_image, min_max_image, calib.cam_K, calib.cam_kc, calib.proj_K, calib.proj_kc,
                                   calib.R, calib.T)
    return pointcloud


if __name__ == '__main__':
    # pattern_file_list = glob.glob('../cartman/2013-May-14_20.41.56.117/*.png')
    from calibrate import CalibrationData

    # pattern_file_list = glob.glob('../data/Nikon/test1/*.JPG')
    # pattern_file_list.sort()
    # count = len(pattern_file_list)
    # out = reconstruct_model_simple(pattern_file_list)
    # np.save('./test_out.npy', out)
    out = np.load("./test_bearing_2.npy")
    out[out[:, :, 2] > 1000] = 0
    out[out[:, :, 2] < 100] = 0
    # out[out<1000] = 0
    xyz = np.zeros((out.shape[0] * out.shape[1], 3), dtype=np.float32)

    xyz[:, 0] = out[:, :, 1].flatten()
    xyz[:, 1] = out[:, :, 0].flatten()
    xyz[:, 2] = out[:, :, 2].flatten()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("./test_bearing_2.ply", pcd)
