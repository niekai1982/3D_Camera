import numpy as np
import os
import glob
import matplotlib as mpl
import time

mpl.use("tkagg")
import matplotlib.pyplot as plt
import cv2
import numba as nb

INVAILID_PIXEL = 0xffff
EMPTY_PIXEL = -1


class GrayCode(object):
    def __init__(self, resolution):
        self.resolution_ = resolution
        self.maximum_patterns_ = np.uint32(np.ceil(np.log2(self.resolution_)))
        self.maximum_disparity_ = np.uint32(1) << self.maximum_patterns_
        self.msb_pattern_value_ = np.uint32(self.maximum_disparity_) >> 1
        self.offset_ = (self.maximum_disparity_ - self.resolution_) / 2
        self.pixel_threshold_ = 10.

    def Setup(self):
        pass

    def GeneratePatternSequence(self):
        pass

    def DecodeCaptureSequence(self, images_coded):
        # pattern_loop_count  = images_coded.shape[0]
        pattern_loop_count = (images_coded.shape[0] - 2) // 2
        image_rows = images_coded.shape[1]
        image_cols = images_coded.shape[2]

        # self.disparity_map_ = -1 * np.ones(shape=(image_rows, image_cols), dtype=np.int32)
        # self.image_albedo_ = np.zeros(shape=(image_rows, image_cols), dtype=np.uint8)

        image_max = images_coded[0].copy()
        image_min = images_coded[1].copy()

        self.image_albedo_ = np.where(image_max >= image_min + self.pixel_threshold_, (image_max + image_min) / 2, 255)
        self.disparity_map_ = np.where(image_max >= (image_min + self.pixel_threshold_), EMPTY_PIXEL, INVAILID_PIXEL)

        pattern_value = self.msb_pattern_value_

        for i in range(pattern_loop_count):
            image_normal = images_coded[i*2+2]
            image_inverted = images_coded[i*2+1+2]
            image_difference = image_normal.astype(np.int32) - image_inverted.astype(np.int32)
            self.disparity_map_ = get_pattern_value(self.disparity_map_, image_difference, pattern_value, threshold=2)
            pattern_value = pattern_value >> 1

    def GetSetup(self):
        pass


@nb.jit(nopython=True)
def get_pattern_value(disparity_map_, image_difference, pattern_value, threshold):
    for row in range(image_difference.shape[0]):
        for col in range(image_difference.shape[1]):
            disparity_value = disparity_map_[row, col]
            if disparity_value != 0xffff:
                if disparity_value == -1:
                    disparity_value = 0
                difference = image_difference[row, col]
                if difference > 0:
                    pixel_pattern_code = pattern_value
                else:
                    pixel_pattern_code = 0
                    difference = -1 * difference
                if difference >= threshold:
                    disparity_value = disparity_value | (
                            pixel_pattern_code ^ ((disparity_value >> 1) & pattern_value))
                else:
                    disparity_value = 0xffff
            disparity_map_[row, col] = disparity_value
    return disparity_map_


if __name__ == '__main__':
    data_path = '/Users/kainie/Downloads/scan_calib_3d_camera/data/TI/11-18-1/*.jpg'
    
    proj_on = cv2.imread(
        '/Users/kainie/Downloads/scan_calib_3d_camera/data/TI/11-18-1/LUCID_TRI032S-C_212401114__20211118094415877_image702.jpg',
        0)
    proj_off = cv2.imread(
        '/Users/kainie/Downloads/scan_calib_3d_camera/data/TI/11-18-1/LUCID_TRI032S-C_212401114__20211118094415970_image703.jpg',
        0)

    proj_coded = np.stack((proj_on, proj_off), axis=0)
    
    images_set = glob.glob(data_path)
    images_set.sort()
    
    images_coded_h = np.array([cv2.imread(fn, 0) for fn in images_set[:22]])
    images_coded_v = np.array([cv2.imread(fn, 0) for fn in images_set[24:44]])

    graycode_h = GrayCode(1280)
    graycode_v = GrayCode(720)

    start = time.time()
    graycode_h.DecodeCaptureSequence(np.vstack((proj_coded, images_coded_h)))
    graycode_v.DecodeCaptureSequence(np.vstack((proj_coded, images_coded_v)))
    print("Decode Spend Time:", time.time() - start)


    plt.subplot(221)
    plt.imshow(proj_on)
    plt.subplot(222)
    plt.imshow(graycode_h.image_albedo_)
    plt.subplot(223)
    plt.imshow(np.where(graycode_h.disparity_map_==INVAILID_PIXEL, 0, graycode_h.disparity_map_))
    plt.subplot(224)
    plt.imshow(np.where(graycode_v.disparity_map_==INVAILID_PIXEL, 0, graycode_v.disparity_map_))
    plt.show()

    pass
