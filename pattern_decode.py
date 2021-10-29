import cv2
import numpy as np
import os
import glob
import matplotlib as mpl

mpl.use('tkagg')
import matplotlib.pyplot as plt


class cvSize(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h


BIT_UNCERTAIN = 0xffff
PIXEL_UNCERTAIN = np.nan
gray_offset = (0, 128)
projector_size = cvSize(1600, 1200)
RobustDecode = 0x01
GrayPatternDecode = 0x02
DEFAULT_B = 0.3
DEFAULT_M = 5


def INVALID(value):
    return np.isnan(value)


def util_grayToBinary(num, numBits):
    shift = 1
    while shift < numBits:
        num ^= num >> shift
        shift = shift << 1
    return num


def util_binaryToGray(num):
    return (num >> 1) ^ num


def grayToBinary(value, set):
    return util_grayToBinary(value, 32) - gray_offset[set]


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
    direct_light[0] = np.where(Lg > 0, Ld, Lmax)
    direct_light[1] = np.where(Lg > 0, Lg, 0)
    return direct_light


def get_robust_bit(value1, value2, Ld, Lg, m):
    if Ld < m:
        return BIT_UNCERTAIN
    if Ld > Lg:
        if value1 > value2:
            return 1
        else:
            return 0
    if value1 <= Ld and value2 >= Lg:
        return 0
    if value1 >= Lg and value2 <= Ld:
        return 1
    return BIT_UNCERTAIN


def convert_pattern(pattern_image, binary):
    if pattern_image.shape[0] == 0:
        return
    if pattern_image.shape[2] < 2:
        return
    if binary:
        print("Converting binary code to gray")
    else:
        print("Converting gray code to binary")

    def b2g_proc(pattern, function):
        if not np.isnan(pattern):
            p = int(pattern)
            res = function(p, 0) + pattern - p
        else:
            res = pattern
        return res

    def g2b_proc(pattern, function, threshold):
        if not np.isnan(pattern):
            p = int(pattern)
            code = function(p, 0)
            if code < 0:
                code = 0
            elif code >= threshold:
                code = threshold - 1
            res = code + pattern - p
        else:
            res = pattern
        return res

    if binary:
        pattern0 = np.vectorize(b2g_proc)(pattern_image[:, :, 0], binaryToGray)
        pattern1 = np.vectorize(b2g_proc)(pattern_image[:, :, 1], binaryToGray)
    else:
        pattern0 = np.vectorize(g2b_proc)(pattern_image[:, :, 0], grayToBinary, projector_size.width)
        pattern1 = np.vectorize(g2b_proc)(pattern_image[:, :, 1], grayToBinary, projector_size.height)

    pattern_out = np.dstack((pattern0, pattern1))
    return pattern_out


def decode_pattern(pattern_image_list, flag, direct_light, m):
    binary = (flag & GrayPatternDecode) != GrayPatternDecode
    robust = (flag & RobustDecode) == RobustDecode

    # inline function
    def get_min(init, value1, value2, min):
        if init or value1 < min or value2 < min:
            res = value1 if value1 < value2 else value2
        else:
            res = min
        return res

    def get_max(init, value1, value2, max):
        if init or value1 > max or value2 > max:
            res = value1 if value1 > value2 else value2
        else:
            res = max
        return res

    images = []
    for idx in range(len(pattern_image_list)):
        images.append(cv2.imread(pattern_image_list[idx], 0))
    images = np.array(images)

    print("--- decode_pattern START ---")
    init = True

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

        min_max_image[:, :, 0] = np.vectorize(get_min)(init, gray_image1, gray_image2, min_max_image[:, :, 0])
        min_max_image[:, :, 1] = np.vectorize(get_max)(init, gray_image1, gray_image2, min_max_image[:, :, 1])

        if not robust:
            pattern_image[:, :, channel][np.where(gray_image1 > gray_image2)] += (1 << bit)
        else:
            p = np.vectorize(get_robust_bit)(gray_image1, gray_image2, direct_light[0], direct_light[1], m)

            def p2pattern(p, pattern):
                if p == BIT_UNCERTAIN:
                    return PIXEL_UNCERTAIN
                else:
                    return pattern + (int(p) << bit)

            pattern_image[:, :, channel] = np.vectorize(p2pattern)(p, pattern_image[:, :, channel])

        init = False
        current += 1
    if not binary:
        pattern_image = convert_pattern(pattern_image, binary)

    print("--- decode_pattern END ---")
    return pattern_image, min_max_image


def decode_gray_set(pattern_image_list):
    direct_component_images = [15, 16, 17, 18, 35, 36, 37, 38]
    # direct_component_images = [15, 16, 17, 18, 30, 31, 32, 33]
    # direct_component_images = [5, 6, 7, 8, 11, 12, 13, 14]
    images = []
    for idx in direct_component_images:
        images.append(cv2.imread(pattern_image_list[idx - 1], 0))
    images = np.array(images)
    direct_light = estimate_direct_light(images, b=0.5)
    pattern_image, min_max_image = decode_pattern(pattern_image_list, RobustDecode | GrayPatternDecode, direct_light,
                                                  DEFAULT_M)
    return pattern_image, min_max_image


def decode():
    pass


if __name__ == "__main__":
    # pattern_file_list = glob.glob('../data/cartman/2013-May-14_20.41.56.117/*.png')
    # pattern_file_list = glob.glob('../cartman/2013-May-14_20.41.56.117/*.png')
    pattern_file_list = glob.glob('../data/Nikon/3/*.JPG')
    pattern_file_list.sort()
    count = len(pattern_file_list)
    pattern_image, min_max_image = decode_gray_set(pattern_file_list)
    np.save("Nikon_3_pattern_image.npy", pattern_image)
    np.save("Nikon_3_min_max_image.npy", min_max_image)
    # plt.subplots(221)
    # plt.imshow(pattern_image[:, :, 0])
    # plt.subplots(222)
    # plt.imshow(pattern_image[:, :, 1])
    # plt.subplots(223)
    # plt.imshow(min_max_image[:, :, 0])
    # plt.subplots(224)
    # plt.imshow(min_max_image[:, :, 1])
    # plt.show()
