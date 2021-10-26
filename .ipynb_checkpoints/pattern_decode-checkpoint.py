import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    pass