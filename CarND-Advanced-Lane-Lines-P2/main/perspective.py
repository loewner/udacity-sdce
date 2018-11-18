import cv2
import numpy as np

class Perspective(object):
    def __init__(self, img_path, src, offset = 300):
        self.img_path = img_path
        self.src = src
        self.img = cv2.imread(img_path)
        im_shape = self.img.shape
        center = int(im_shape[1] / 2)

        self.dst = np.float32([[center - offset, im_shape[0]], [center - offset, 0], [center + offset, 0],
                          [center + offset, im_shape[0]]])

        self.M = cv2.getPerspectiveTransform(np.float32([self.src]), self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, np.float32([self.src]))

    def apply(self, img):
        im_shape = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M, im_shape)