import numpy as np
import os
import cv2
import open3d as o3d


class Point(object):
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.distances = 0.