import cv2
import os
import numpy as np
import matplotlib as mpl
mpl.use("tkagg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('/Volumes/Untitled/DaHeng_camera_image/calibration_image_1217/001-1.bmp')
    ret, centers = cv2.findCirclesGrid(img[:,:,0], (7,7))
    plt.imshow(img)
    plt.scatter(centers[:,0,0], centers[:,0,1],s=0.1,c='r')
    plt.show()