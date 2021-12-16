import cv2
import numpy as np
import os
import glob
import matplotlib as mpl
from calibrate import CalibrationData
from cv2python import *
from common.point_cloud import Point

mpl.use("tkagg")
import matplotlib.pyplot as plt
import numba as nb

projector_size = cvSize(1280, 720)
camera_size = cvSize(2048, 1536)

calib = CalibrationData(camera_resolution=[1, 1], projector_resolution=[1280, 720])
calib.cam_K = np.array([[4.55870035e+03, 0.00000000e+00, 1.51017862e+03],
                        [0.00000000e+00, 4.55911080e+03, 9.88716486e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

calib.cam_kc = np.array([[-1.95063253e-01],
                         [1.93858801e+00],
                         [-4.99197011e-04],
                         [1.35514749e-03],
                         [-1.55614418e+01]], dtype=np.float32)

calib.proj_K = np.array([[3.90467278e+03, 0.00000000e+00, 8.02215392e+02],
                         [0.00000000e+00, 3.90997537e+03, 6.01700509e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

calib.proj_kc = np.array([[-4.78490045e-01],
                          [3.86902030e+01],
                          [-8.60596372e-02],
                          [1.31578421e-02],
                          [-6.33953261e+02]], dtype=np.float32)

calib.R = np.array([[0.93488111, -0.06929929, 0.34813061],
                    [-0.02243265, 0.9672622, 0.25278571],
                    [-0.35425145, -0.24413408, 0.90272059]], dtype=np.float32)

calib.T = np.array([[-319.41056464],
                    [-230.99720084],
                    [525.08697085]], dtype=np.float32)
calib.is_valid = True


class PlaneEqation(object):
    def __init__(self, x=0, y=0, z=0, d=0):
        self.w = cvPoint3d(x, y, z)
        self.d = 0

    def GetA(self):
        return self.w.x

    def GetB(self):
        return self.w.y

    def GetC(self):
        return self.w.z


class ViewPoint(object):
    def __init__(self):
        self.center = cvPoint3d()
        self.ray = []
        self.plane_columns = []
        self.plane_rows = []
        self.plane_diamond_angle_1 = []
        self.plane_diamond_angle_2 = []


class Geometry(object):
    def __init__(self):
        self.origin_ = ViewPoint()
        self.is_setup_ = False
        self.origin_set_ = False
        self.viewport_ = []

    def GenerateOpticalPoints(self, columns, rows, compress_x, compress_y, shift_rows, shift_colums, over_sample_colums, \
                              over_sample_rows, intrinsics, distortion):
        # distortion[4] = 0
        total_columns = columns * over_sample_colums
        total_rows = rows * over_sample_rows

        distortion[:] = 0
        ray_count = total_columns * total_rows
        points = np.mgrid[:total_rows, :total_columns].astype(np.float64)
        # undistorted_points = np.mgrid[:total_rows, :total_columns].astype(np.float64)
        points = np.dstack((points[0].flatten(), points[1].flatten())).reshape(-1, 1, 2)
        # undistorted_points = np.dstack((undistorted_points[0].flatten(), undistorted_points[1].flatten())).reshape(-1,2)
        undistorted_points = cv2.undistortPoints(points, intrinsics, distortion)
        tmp_x = undistorted_points[:, 0, 0] * intrinsics[0, 0] + intrinsics[0, 2]
        tmp_y = undistorted_points[:, 0, 1] * intrinsics[1, 1] + intrinsics[1, 2]
        return ray_count, total_rows, total_columns, points, undistorted_points

    def SetOriginView(self, origin_calib):
        self.origin_calib_ = origin_calib
        self.origin_.center.x = 0.
        self.origin_.center.y = 0.
        self.origin_.center.z = 0.

        columns = projector_size.cols
        rows = projector_size.rows

        ray_count, total_rows, total_columns, _, undistorted_rays = self.GenerateOpticalPoints(columns, rows,
                                                                                               compress_x=1.,
                                                                                               compress_y=1.,
                                                                                               distortion=self.origin_calib_.proj_kc,
                                                                                               intrinsics=self.origin_calib_.proj_K,
                                                                                               over_sample_colums=1,
                                                                                               over_sample_rows=1,
                                                                                               shift_colums=0.,
                                                                                               shift_rows=0.)
        total_angled_lines = total_columns + (total_rows // 2)
        ray_length = np.sqrt(np.power(undistorted_rays[:, 0, 0], 2) + np.power(undistorted_rays[:, 0, 1], 2) + 1)
        ray_normal_x = undistorted_rays[:, 0, 0] / ray_length
        ray_normal_y = undistorted_rays[:, 0, 1] / ray_length
        ray_normal_z = np.ones_like(undistorted_rays[:, 0, 0]) / ray_length

        ray_normal_x.shape = total_rows, total_columns
        ray_normal_y.shape = total_rows, total_columns
        ray_normal_z.shape = total_rows, total_columns

        self.origin_.ray = np.dstack((np.dstack((ray_normal_x, ray_normal_y)), ray_normal_z))

        # create column planes
        for xCol in range(total_columns):
            points = np.empty(shape=(total_rows + 1, 3), dtype=np.float64)
            for yRow in range(total_rows):
                points[yRow, 0] = self.origin_.ray[yRow, xCol, 0]
                points[yRow, 1] = self.origin_.ray[yRow, xCol, 1]
                points[yRow, 2] = self.origin_.ray[yRow, xCol, 2]
            points[total_rows, 0] = self.origin_.center.x
            points[total_rows, 1] = self.origin_.center.y
            points[total_rows, 2] = self.origin_.center.z
            plane_eq = self.FitPlane(points)
            self.origin_.plane_columns.append(plane_eq)
        self.origin_set_ = True

    def AddView(self, viewport_calib):
        viewport_tmp = ViewPoint()

        origin_intrinsic = self.origin_calib_.proj_K
        origin_distortion = self.origin_calib_.proj_kc

        viewport_intrinsic = self.origin_calib_.cam_K
        viewport_distortion = self.origin_calib_.cam_kc

        viewport_origin_R = self.origin_calib_.R
        viewport_origin_T = self.origin_calib_.T

        viewport_tmp.center.x = viewport_origin_T[0]
        viewport_tmp.center.y = viewport_origin_T[1]
        viewport_tmp.center.z = viewport_origin_T[2]

        rows = camera_size.rows
        columns = camera_size.cols

        ray_count = columns * rows

        original_rays = np.mgrid[:rows, :columns].astype(np.float64)
        original_rays.shape = -1, 1, 2

        undistorted_rays = cv2.undistortPoints(original_rays, viewport_intrinsic, viewport_distortion)

        undistorted_rays.shape = rows, columns, 2

        rays_length = np.sqrt(np.power(undistorted_rays[:, :, 0], 2) + np.power(undistorted_rays[:, :, 1], 2))

        viewport_rays = np.empty(shape=(rows, columns, 3), dtype=np.float64)

        viewport_rays[:, :, 0] = undistorted_rays[:, :, 0] / rays_length
        viewport_rays[:, :, 1] = undistorted_rays[:, :, 1] / rays_length
        viewport_rays[:, :, 2] = np.ones((rows, columns), dtype=np.float64) / rays_length

        viewport_rays_tmp = viewport_rays.copy()

        viewport_rays_tmp.shape = -1, 3

        viewport_rays_tmp = viewport_origin_R * viewport_rays_tmp

        viewport_rays_tmp.shape = rows, columns, 3

        viewport_tmp.ray = viewport_rays_tmp.copy()

        self.viewport_.append(viewport_tmp)



    def FitPlane(self, points):
        plane_eq = PlaneEqation()
        covariance, centroid = cv2.calcCovarMatrix(points, mean=0, flags=cv2.COVAR_ROWS | cv2.COVAR_NORMAL)
        w, u, vt = cv2.SVDecomp(covariance)
        plane_eq.w.x = vt[2, 0]
        plane_eq.w.y = vt[2, 1]
        plane_eq.w.z = vt[2, 2]

        plane_eq.d = 0.
        plane_eq.d = plane_eq.d + plane_eq.w.x * centroid[0, 1]
        plane_eq.d = plane_eq.d + plane_eq.w.y * centroid[0, 1]
        plane_eq.d = plane_eq.d + plane_eq.w.z * centroid[0, 1]

        return plane_eq

    def Unsafe_Find3dPlaneLineIntersection(self, origin_plan, orientation, viewport_id, viewpoint_x, viewpoint_y):
        plane_eq = self.origin_.plane_columns[origin_plan]
        q = self.viewport_[viewport_id].center
        v = self.viewport_[viewport_id].ray[viewport_id, viewpoint_x]

        n_dot_q = np.dot(plane_eq.w, q)
        n_dot_v = np.dot(plane_eq.w, v)

        ret_xyz = Point()
        ret_xyz.distance = (plane_eq.d - n_dot_q) / n_dot_v

        ret_xyz.x = q.x + ret_xyz.distance * v.x
        ret_xyz.y = q.y + ret_xyz.distance * v.y
        ret_xyz.z = q.z + ret_xyz.distance * v.z

        return ret_xyz

    def GeneratePointCloud(self, viewport_id, disparty_map):
        point_cloud = []
        depth_map = []
        return point_cloud, depth_map


if __name__ == '__main__':
    geometry = Geometry()
    # out = geometry.GenerateOpticalPoints(camera_size.cols, camera_size.rows, compress_x=1.0, compress_y=1.0,
    #                                      shift_rows=0.0, shift_colums=0.0, over_sample_colums=1,
    #                                      over_sample_rows=1, intrinsics=calib.cam_K, distortion=calib.cam_kc)
    # out = geometry.GenerateOpticalPoints(projector_size.cols, projector_size.rows, compress_x=1.0, compress_y=1.0,
    #                                      shift_rows=0.0, shift_colums=0.0, over_sample_colums=1,
    #                                      over_sample_rows=1, intrinsics=calib.proj_K, distortion=calib.proj_kc)
    out = geometry.SetOriginView(calib)
