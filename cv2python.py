import cv2
import numpy as np

class cvSize(object):
    def __init__(self, w=0, h=0):
        self.width = w
        self.height = h
        self.cols = w
        self.rows = h

class cvPoint3d(object):
    def __init__(self,x=0.,y=0.,z=0.):
        self.x = x
        self.y = y
        self.z = z
