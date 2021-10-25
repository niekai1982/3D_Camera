import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

class cvSize(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h


BIT_UNCERTAIN = 65535
gray_offset = (0, 128)
projector_size = cvSize(1024, 768)
RobustDecode = 0x01
GrayPatternDecode = 0x02
DEFAULT_B = 0.3
DEFAULT_M = 5


def INVALID(value):
    return np.isnan(value)

def util_grayToBinary(num, numBits):
    shift = 1
    while shift<numBits:
        num ^= num >> shift
        shift = shift << 1
    return num

def util_binaryToGray(num):
    return (num>>1) ^ num

def grayToBinary(value, set):
    return util_grayToBinary(value, 32) - gray_offset[set]

def binaryToGray(value, set):
    return util_binaryToGray(value) - gray_offset[set]

def estimate_direct_light(images, b):
    count = images.shape[0]
    if count < 1:
        return []
    direct_light = np.empty((2, images.shape[1], images.shape[2]))
    b1 = 1./(1. - b)
    b2 = 2./(1. - b * 1. * b)
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
        p = int(pattern)
        res = function(p, 0) + pattern - p
        return res
    def g2b_proc(pattern, function, threshold):
        p = int(pattern)
        code = function(p, 0)
        if code < 0:
            code = 0
        elif code >= threshold:
            code = threshold - 1
        res = code + pattern - p
        return res
    if binary:
        pattern0 = np.vectorize(b2g_proc)(pattern_image[:,:,0], binaryToGray)
        pattern1 = np.vectorize(b2g_proc)(pattern_image[:,:,1], binaryToGray)
    else:
        pattern0 = np.vectorize(g2b_proc)(pattern_image[:,:,0], grayToBinary, projector_size.width)

        pattern1 = np.vectorize(g2b_proc)(pattern_image[:,:,1], grayToBinary, projector_size.height)
    pattern_out = np.dstack((pattern0, pattern1))
    return pattern_out

def decode_pattern(pattern_image_list, flag, direct_light, m):

    binary = (flag & GrayPatternDecode)!=GrayPatternDecode
    robust = (flag & RobustDecode) == RobustDecode

    images = []
    for idx in range(len(pattern_image_list)):
        images.append(cv2.imread(pattern_image_list[idx]))
    images = np.array(images)

    print("--- decode_pattern START ---")
    init = True

    total_images = images.shape[0]
    total_patterns = total_images/2 - 1
    total_bits = total_patterns/2


    if (2+4*total_bits) != total_images:
        print("ERROR: cannot detect pattern and bit count from image set.")
        return False
    bit_count = (0, total_bits, total_bits)
    set_size = (1, total_bits, total_bits)
    COUNT = 2*(set_size[0]+set_size[1]+set_size[2])
    pattern_offset = (
        ((1 << total_bits)-projector_size[1])/2, ((1 << total_bits)-projector_size[0])/2)
    if images.shape[0] < COUNT:
        print("Image list size does not match set size")
        return False
    set_idx = 0
    current = 0
    for t in range(0, COUNT, 2):
        current += 1
        if current == set_size[set_idx]:
            set_idx += 1
            current = 0
        if set_idx == 0:
            current += 1
            continue
        bit = bit_count[set_idx] - current - 1
        channel = set_idx - 1
        gray_image1 = images[t+0]
        gray_image2 = images[t+1]

        if init:
            if robust and gray_image1.shape[0] != direct_light.shape[0]:
                print("--> Direct Componect image has different size: ")
                return False
            pattern_image = np.empty((gray_image1.shape[0], gray_image1.shape[1], 2), dtype=np.float32)
            min_max_image = np.empty((gray_image1.shape[0], gray_image1.shape[1], 2), dtype=np.uint8)
        if init:
            pattern_image[:,:,:] = 0.

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

        min_max_image[:,:,0] = np.vectorize

        if  not robust:
            pattern_image[:,:,channel][np.where(gray_image1>gray_image2)] += (1<<bit)



        current += 1


def decode_gray_set(pattern_image_list):
    direct_component_images = [15, 16, 17, 18, 35, 36, 37, 38]
    images = []
    for idx in direct_component_images:
        images.append(cv2.imread(pattern_image_list[idx-1], 0))
    images = np.array(images)
    direct_light = estimate_direct_light(images, b=0.5)
    decode_pattern(pattern_image_list, RobustDecode | GrayPatternDecode, direct_light, DEFAULT_M)
    return direct_light

def decode():
    pass


if __name__ == "__main__":
    pattern_file_list = glob.glob('../3d_camera_calib_data/cartman/2013-May-14_20.41.56.117/*.png')
    pattern_file_list.sort()
    count = len(pattern_file_list)

    out = decode_gray_set(pattern_file_list)
    print(out.shape)
    plt.imshow(out[1])
    plt.show()

