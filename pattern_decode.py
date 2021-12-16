import cv2
import numpy as np
import os
import glob
import matplotlib as mpl
import numba as nb
from numba import njit
from numba import float32, float64, int8
from numba import bool_
import time

# mpl.use('tkagg')
import matplotlib.pyplot as plt


class cvSize(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h


BIT_UNCERTAIN = 0xffff
PIXEL_UNCERTAIN = np.nan
gray_offset = (0, 128)
projector_size = cvSize(1280, 720)
RobustDecode = 0x01
GrayPatternDecode = 0x02
DEFAULT_B = 0.3
DEFAULT_M = 5


def INVALID(value):
    return np.isnan(value)


@nb.jit(nopython=True)
def util_grayToBinary(num, numBits):
    shift = 1
    while shift < numBits:
        num ^= num >> shift
        shift = shift << 1
    return num


@nb.jit(nopython=True)
def util_binaryToGray(num):
    return (num >> 1) ^ num


@nb.jit(nopython=True)
def grayToBinary(value, set):
    return util_grayToBinary(value, 32) - gray_offset[set]


@nb.jit(nopython=True)
def binaryToGray(value, set):
    return util_binaryToGray(value) - gray_offset[set]


def estimate_direct_light(images, b):
    count = images.shape[0]
    if count < 1:
        return []
    direct_light = np.empty((2, images.shape[1], images.shape[2]))
    b1 = 1. / (1. - b)
    b2 = 2. / (1. - b * 1. * b)
    Lmax = np.max(images, axis=0)
    Lmin = np.min(images, axis=0)
    Ld = b1 * (Lmax - Lmin) + 0.5
    Lg = b2 * (Lmin - b * Lmax) + 0.5
    # plt.imshow(Lg)
    # plt.colorbar()
    # plt.show()
    direct_light[0] = np.where(Lg > 0, Ld, Lmax)
    direct_light[1] = np.where(Lg > 0, Lg, 0)
    return direct_light


@nb.jit(nopython=True)
def get_robust_bit(a, b, Ld, Lg, m):
    h, w = a.shape
    res_array = np.empty(shape=(h, w))
    for i in range(h):
        for j in range(w):
            if Ld[i, j] < m:
                res_array[i, j] = BIT_UNCERTAIN
            elif Ld[i, j] > Lg[i, j]:
                if a[i, j] > b[i, j]:
                    res_array[i, j] = 1
                else:
                    res_array[i, j] = 0
            elif a[i, j] <= Ld[i, j] and b[i, j] >= Lg[i, j]:
                res_array[i, j] = 0
            elif a[i, j] >= Lg[i, j] and b[i, j] <= Ld[i, j]:
                res_array[i, j] = 1
            else:
                res_array[i, j] = BIT_UNCERTAIN
    return res_array


@nb.jit(nopython=True)
def b2g_proc(pattern, function):
    h, w = pattern.shape
    res_array = np.empty_like(pattern)
    for i in range(h):
        for j in range(w):
            p_float = pattern[i, j]
            if not np.isnan(p_float):
                p_int = int(p_float)
                res_array[i, j] = function(p_int, 0) + p_float - p_int
            else:
                res_array[i, j] = p_float
    return res_array


@nb.jit(nopython=True)
def g2b_proc(pattern, function, threshold):
    h, w = pattern.shape
    res_array = np.empty_like(pattern)
    for i in range(h):
        for j in range(w):
            p_float = pattern[i, j]
            if not np.isnan(p_float):
                p_int = int(p_float)
                code = function(p_int, 0)
                if code < 0:
                    code = 0
                elif code >= threshold:
                    code = threshold - 1
                res_array[i, j] = code + p_float - p_int
            else:
                res_array[i, j] = p_float
    return res_array


@nb.jit(nopython=True)
def convert_pattern_b2g(pattern_image, projector_size):
    pattern0 = b2g_proc(pattern_image[:, :, 0], binaryToGray)
    pattern1 = b2g_proc(pattern_image[:, :, 1], binaryToGray)
    pattern_out = np.dstack((pattern0, pattern1))
    return pattern_out


@nb.jit(nopython=True)
def convert_pattern_g2b(pattern_image, projector_size_width, projector_size_height):
    pattern0 = g2b_proc(pattern_image[:, :, 0], grayToBinary, projector_size_width)
    pattern1 = g2b_proc(pattern_image[:, :, 1], grayToBinary, projector_size_height)
    pattern_out = np.dstack((pattern0, pattern1))
    return pattern_out


# @nb.jit(nppython=True)
def decode_pattern(pattern_image_list, flag, direct_light, m):
    binary = (flag & GrayPatternDecode) != GrayPatternDecode
    robust = (flag & RobustDecode) == RobustDecode
    robust = False

    # inline function
    @nb.jit(nopython=True)
    def get_min(init, a, b, min):
        h, w = a.shape
        res_array = np.empty(shape=(h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                value1 = a[i, j]
                value2 = b[i, j]
                min_value = min[i, j]
                if init or value1 < min_value or value2 < min_value:
                    res_array[i, j] = value1 if value1 < value2 else value2
                else:
                    res_array[i, j] = min_value
        return res_array

    @nb.jit(nopython=True)
    def get_max(init, a, b, max):
        h, w = a.shape
        res_array = np.empty(shape=(h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                value1 = a[i, j]
                value2 = b[i, j]
                max_value = max[i, j]
                if init or value1 > max_value or value2 > max_value:
                    res_array[i, j] = value1 if value1 > value2 else value2
                else:
                    res_array[i, j] = max_value
        return res_array

    images = []
    for idx in range(len(pattern_image_list)):
        images.append(cv2.imread(pattern_image_list[idx], 0))
    images = np.array(images)

    print("--- decode_pattern START ---")
    init = True
    start_1 = time.time()

    total_images = images.shape[0]
    total_patterns = int(total_images / 2 - 1)
    total_bits = int(total_patterns / 2)

    if (2 + 4 * total_bits) != total_images:
        print("ERROR: cannot detect pattern and bit count from image set.")
        return False
    bit_count = (0, total_bits, total_bits)
    set_size = (1, total_bits, total_bits)
    COUNT = 2 * (set_size[0] + set_size[1] + set_size[2])
    pattern_offset = (
        ((1 << total_bits) - projector_size.width) / 2, ((1 << total_bits) - projector_size.height) / 2)
    if images.shape[0] < COUNT:
        print("Image list size does not match set size")
        return False
    set_idx = 0
    current = 0
    for t in range(0, COUNT, 2):
        print(t)
        if current == set_size[set_idx]:
            set_idx += 1
            current = 0
        if set_idx == 0:
            current += 1
            continue
        bit = bit_count[set_idx] - current - 1
        channel = set_idx - 1
        gray_image1 = images[t + 0]
        gray_image2 = images[t + 1]

        if init:
            if robust and gray_image1.shape[0] != direct_light.shape[1]:
                print("--> Direct Componect image has different size: ")
                return False
            pattern_image = np.empty((gray_image1.shape[0], gray_image1.shape[1], 2), dtype=np.float32)
            min_max_image = np.empty((gray_image1.shape[0], gray_image1.shape[1], 2), dtype=np.uint8)
        if init:
            pattern_image[:, :, :] = 0.

        # def robust_pro(init, row_light, pattern):
        #     if row_light and (init or pattern != PIXEL_UNCERTAIN):

        start = time.time()
        min_max_image[:, :, 0] = get_min(init, gray_image1, gray_image2, min_max_image[:, :, 0])
        min_max_image[:, :, 1] = get_max(init, gray_image1, gray_image2, min_max_image[:, :, 1])
        print("get min_max_image spend time:", time.time() - start)

        if not robust:
            pattern_image[:, :, channel][np.where(gray_image1 > gray_image2)] += (1 << bit)
        else:
            start = time.time()
            p = get_robust_bit(gray_image1, gray_image2, direct_light[0], direct_light[1], m)
            print("get robust bit spend time:", time.time() - start)

            # def p2pattern(p, pattern, bit):
            #     if p == BIT_UNCERTAIN:
            #         return PIXEL_UNCERTAIN
            #     else:
            #         return pattern + (int(p) << bit)

            @nb.jit(nopython=True)
            def p2pattern(p, pattern, bit):
                h, w = p.shape
                res_array = np.empty(shape=(h, w))
                for i in range(h):
                    for j in range(w):
                        if p[i, j] == BIT_UNCERTAIN:
                            res_array[i, j] = PIXEL_UNCERTAIN
                        else:
                            res_array[i, j] = pattern[i, j] + (int(p[i, j]) << bit)
                return res_array

            pattern_image[:, :, channel] = p2pattern(p, pattern_image[:, :, channel], bit)
            print("p2pattern spend time:", time.time() - start)

        init = False
        current += 1
    if not binary:
        start = time.time()
        pattern_image = convert_pattern_g2b(pattern_image, projector_size.width, projector_size.height)
        print("conver_pattern spend time:", time.time() - start)
    else:
        start = time.time()
        pattern_image = convert_pattern_b2g(pattern_image, projector_size)
        print("conver_pattern spend time:", time.time() - start)

    print("decode spend time:", time.time() - start_1)
    print("--- decode_pattern END ---")
    return pattern_image, min_max_image


# @nb.jit(nopython=True)
def decode_gray_set(pattern_image_list):
    direct_component_images = [15, 16, 17, 18, 35, 36, 37, 38]
    # direct_component_images = [15, 16, 17, 18, 30, 31, 32, 33]
    # direct_component_images = [5, 6, 7, 8, 11, 12, 13, 14]
    images = []
    for idx in direct_component_images:
        images.append(cv2.imread(pattern_image_list[idx - 1], 0))
    images = np.array(images)

    start = time.time()
    direct_light = estimate_direct_light(images, b=0.8)
    print("direct light spend time: ", (time.time() - start))

    pattern_image, min_max_image = decode_pattern(pattern_image_list, RobustDecode | GrayPatternDecode, direct_light,
                                                  DEFAULT_M)
    return pattern_image, min_max_image

def decode_pattern_ti(pattern_image_list):
    pass


def decode():
    pass


if __name__ == "__main__":
    # pattern_file_list = glob.glob('../data/cartman/2013-May-14_20.41.56.117/*.png')
    # pattern_file_list = glob.glob('../cartman/2013-May-14_20.41.56.117/*.png')
    # for i in range(3, 9):
    pattern_file_list = glob.glob('../data/TI/captureImage20211110/savedimages1/*.jpg')
    pattern_file_list.sort()
    for elem in pattern_file_list:
        print(elem)
    # pattern_file_list = [elem for elem in pattern_file_list_src[:-2]]
    # pattern_file_list.insert(0, pattern_file_list_src[-1])
    # pattern_file_list.insert(0, pattern_file_list_src[-2])
    print(pattern_file_list)
    pattern_file_list_c = [elem for elem in pattern_file_list[:-2]]
    pattern_file_list_c.insert(0, pattern_file_list[-1])
    pattern_file_list_c.insert(0, pattern_file_list[-2])
    for elem in pattern_file_list_c:
        print(elem)

    # print(os.getcwd())
    # print(pattern_file_list)
    # pattern_file_list.sort()
    count = len(pattern_file_list_c)
    pattern_image, min_max_image = decode_gray_set(pattern_file_list_c)
    # np.save("Nikon_" + str(i+1) + "_pattern_image.npy", pattern_image)
    # np.save("Nikon_" + str(i+1) + "_min_max_image.npy", min_max_image)
    # np.save("Nikon_test1_pattern_image.npy", pattern_image)
    # np.save("Nikon_test1_min_max_image.npy", min_max_image)
    # plt.subplots(221)
    # plt.imshow(pattern_image[:, :, 0])
    # plt.subplots(222)
    # plt.imshow(pattern_image[:, :, 1])
    # plt.subplots(223)
    # plt.imshow(min_max_image[:, :, 0])
    # plt.subplots(224)
    # plt.imshow(min_max_image[:, :, 1])
    # plt.show()
