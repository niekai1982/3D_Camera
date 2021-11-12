import cv2
import numpy as np
import matplotlib as mpl
mpl.use("tkagg")
import matplotlib.pyplot as plt
import os
import time
import numba as nb


# @nb.jit(nopython=True)
def get_min(init, value1, value2, min):
    if init or value1 < min or value2 < min:
        res = value1 if value1 < value2 else value2
    else:
        res = min
    return res

@nb.jit(nopython=True)
def get_min_a(init, a, b, c):
    h, w = a.shape
    res_array = np.empty(shape=(h, w))
    for i in range(h):
        for j in range(w):
            value1 = a[i,j]
            value2 = b[i, j]
            min = c[i, j]
            if init or value1 < min or value2 < min:
                res_array[i, j] = value1 if value1 < value2 else value2
            else:
                res_array[i, j] = min
    return res_array

@nb.jit(nopython=True)
def get_max(init, value1, value2, max):
    if init or value1 > max or value2 > max:
        res = value1 if value1 > value2 else value2
    else:
        res = max
    return res

if __name__ == "__main__":
    a = np.random.randn(2008,3008)
    b = np.random.randn(2008,3008)
    c = np.random.randn(2008,3008)
    # print(a.shape)
    start = time.time()
    d = np.vectorize(get_min)(True, a, b, c)
    print("vectorize spend time:", time.time() - start)

    start = time.time()
    e = get_min_a(True, a, b, c)
    print("init spend time:", time.time() - start)

    plt.imshow(e-d)
    plt.show()
