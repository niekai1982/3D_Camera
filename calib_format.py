import cv2
import numpy as np
import os

pattern_size = (9, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= 10

print(pattern_points.shape)

corners = np.random.randn(pattern_points.shape[0],2)
corners = np.float32(corners)

print(corners.shape)

criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 50)
# cal_flag = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
cal_flag = 0 + cv2.CALIB_FIX_K3
cal_flag += cv2.CALIB_ZERO_TANGENT_DIST

objectPoints = []
imagePoints = []

objectPoints.append(pattern_points)
objectPoints.append(pattern_points)

imagePoints.append(corners)
imagePoints.append(corners)

cam_ret, cam_k, cam_kc, cam_rvecs, cam_tvecs = cv2.calibrateCamera(objectPoints, imagePoints, (100, 100), cal_flag, criteria)
