import numpy as np

class Params(object):
    def __init__(self):
        # paramter for source bounding box to do perspective transform
        self.points=np.array([[204, 720], [595, 450], [684, 450], [1104, 720]])
        #self.points = np.array([[201, 720], [565, 470], [717, 470], [1112, 720]])
        #self.points = np.array([[202, 720], [585, 460], [698, 460], [1110, 720]])


        # filter yellow
        self.yellow_h_thresh = (20, 140)
        self.yellow_l_thresh = (100, 255)
        self.yellow_s_thresh = (100, 255)

        # filter white
        self.white_l_thresh = (150, 255)

        # sobel s
        self.s_thresh_mag = (100, 255)
        self.s_thresh_dir = (0.7, 1.3)

        # sobel l
        self.l_thresh_mag = (100, 255)
        self.l_thresh_dir = (0.7, 1.3)

        self.sobel_kernel = 3

        # find lanes in binary
        self.margin = 70
        self.nwindows = 9
        self.minpix = 50

        # number of images that have an influence to videos
        self.n = 2

        self.ym_per_pix = 40 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 600  # meters per pixel in x dimension



