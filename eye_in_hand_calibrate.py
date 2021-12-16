import cv2
import os
import glob
import numpy as np
import matplotlib as mpl

mpl.use("tkagg")
import matplotlib.pyplot as plt

base_coor = [1885.87, 1454.31]
base_angle = 1.32

robot_base_point = [-4.43776941, 533.394226, 382.743958]

# x, y, z, A, B, C
robot_points = [[16.7271461, 522.863, 382.743958, -134.037704, 0.840482533, 179.225723],
                [6.72714663, 522.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [-3.27285337, 522.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [-13.2728539, 522.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [-13.2728529, 532.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [-3.27285337, 532.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [6.72714663, 532.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [16.7271461, 532.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [16.7271461, 542.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [6.72714663, 542.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [-3.27285314, 542.863, 382.743958, -134.037704, 0.840482593, 179.225723],
                [-13.2728529, 542.863, 382.743958, -134.037704, 0.840482593, 179.225723]]

# image_points = [[1146, 364],
#                 [1513, 369],
#                 [1879, 376],
#                 [2241, 378],
#                 [2237, 746],
#                 [1874, 740],
#                 [1509, 735],
#                 [1144, 731],
#                 [1140, 1096],
#                 [1504, 1099],
#                 [1868, 1104],
#                 [2233, 1111]]

image_points = [[1118.78, 1062.75],
                [1484.8, 1065.51],
                [1849.83, 1069.76],
                [2213.39, 1073.85],
                [2207.45, 1442.17],
                [1845.13, 1436.1],
                [1480.15, 1433.08],
                [1114.93, 1429.88],
                [1111.69, 1795.04],
                [1476.55, 1798.23],
                [1841.31, 1802.14],
                [2203.86, 1805.81]]

# test_image_point = [[1914,757]]
# test_image_point = [[1920,676]]
# test_image_point = [1474.6, 1378.37]
# test_angle = 62.78

test_image_point = [2323.14,1230.88]
test_angle = 65.99

test_image_point = [2095.04,1337.35]
test_angle = 10.92

test_image_point = [1999.33,1418.28]
test_angle = 1.29

if __name__ == "__main__":
    image_set = glob.glob('/users/kainie/Documents/smart_assembling/image_10mm_211124/*.bmp')
    image_set.sort()

    image_points_uv = np.array(image_points, dtype=np.float32)
    robot_points_xy = np.array(robot_points, dtype=np.float32)[:, :2]

    image_points_uv.shape = -1, 1, 2
    robot_points_xy.shape = -1, 1, 2

    M, mask = cv2.findHomography(image_points_uv, robot_points_xy, cv2.RANSAC, 5.0)

    test_point = np.array(test_image_point, dtype=np.float32)
    test_point.shape = -1, 1, 2
    # out_robot_point = cv2.perspective(test_point, M)
    out_robot_point = cv2.perspectiveTransform(test_point, M)

    deta_x = out_robot_point[:, :, 0] - robot_base_point[0]
    deta_y = out_robot_point[:, :, 1] - robot_base_point[1]
    deta_angle = test_angle - base_angle

    print("deta_x:", -1 * deta_x)
    print("deta_y:", -1 * deta_y)
    print("deta_angle:", -1 * deta_angle)

    # for fname in image_set:
    #     img = cv2.imread(fname, 0)
    #     plt.imshow(img)
    #     plt.show()
    # image_set = image_set[:-3]

    # scale = 4
    #
    # for f_name in image_set:
    #     # gray = cv2.imread(image_set[0], 0)
    #     gray = cv2.imread(f_name, 0)
    #     # print(gray.shape)
    #     # plt.imshow(gray)
    #     # plt.show()
    #
    #     gray_resize = cv2.resize(gray, (int(gray.shape[1] / scale), int(gray.shape[0] / scale)))
    #     vis = cv2.cvtColor(gray_resize, cv2.COLOR_GRAY2BGR)
    #     gray_resize = np.float32(gray_resize)
    #     dst = cv2.cornerHarris(gray_resize, 2, 3, 0.04)
    #     dst = cv2.dilate(dst, None)
    #     vis[221:449, 383:632][dst[221:449, 383:632] > 0.2 * dst.max()] = [0, 0, 255]
    #     cv2.imshow("res", vis)
    #     while 1:
    #         ch = cv2.waitKey(1)
    #         if ch == ord("n"):
    #             break
    # cv2.destroyAllWindows()
    # # cv2.destroyWindow()
    #
    # # fast = cv2.FastFeatureDetector()
    # # kp = fast.detect(gray_resize, None)
    # # img2 = cv2.drawKeypoints(gray_resize, kp, color=(255,0,0))
    # # print(img2.shape[0])
    # # plt.imshow(img2)
    # # plt.show()
