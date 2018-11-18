#!/usr/bin/python

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

class Calibrate(object):
    # initialize Simulation object
    def __init__(self, input_folder, nx, ny):
        self.input_folder = input_folder

        self.nx = nx  # the number of inside corners in x
        self.ny = ny  # the number of inside corners in y

        # Make a list of calibration images
        self.images = glob.glob(self.input_folder)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.ny * self.nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(self.images):
            img = cv2.imread(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret==False:
                print("failed to find corners in image " + fname)
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

        test_img = cv2.imread(self.images[0])
        img_size = (test_img.shape[1], test_img.shape[0])

        # Do camera calibration given object points and image points
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)

    def get_images_list(self):
        return self.images

    def undistort(self, img):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst

    def corners_unwarp(self, img):
        # Pass in your image into this function
        # Write code to do the following steps
        # 1) Undistort using mtx and dist
        un_dist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        im_shape = (un_dist.shape[1], un_dist.shape[0])
        offsetx=200
        offsety=100
        # 2) Convert to grayscale
        gray = cv2.cvtColor(un_dist, cv2.COLOR_RGB2GRAY)
        # 3) Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

        # 4) If corners found:
        if ret == True:
            # a) draw corners
            cv2.drawChessboardCorners(un_dist, (self.nx, self.ny), corners, ret)
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
            # Note: you could pick any four of the detected corners
            # as long as those four corners define a rectangle
            # One especially smart way to do this would be to use four well-chosen
            # corners that were automatically detected during the undistortion steps
            # We recommend using the automatic detection of corners in your code
            src = np.concatenate((corners[0], corners[self.nx - 1], corners[self.ny * self.nx - 1], corners[(self.ny - 1) * self.nx]))
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            dst = np.float32([[offsetx, offsety], [im_shape[0] - offsetx, offsety], [im_shape[0] - offsetx, im_shape[1] - offsety],[offsetx, im_shape[1] - offsety]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            M = cv2.getPerspectiveTransform(src, dst)

            # e) use cv2.warpPerspective() to warp your image to a top-down view
            warped = cv2.warpPerspective(un_dist, M, im_shape)

            return ret, warped

        else:
            return ret, gray

    def distortionAndTransformPerspective_testFiles(self, output_folder, perspectiveTransform=True):
        for f in self.images:
            img = cv2.imread(f)

            if perspectiveTransform == False:
                outpath = os.path.join(output_folder, "transf_" + os.path.basename(f))
                cv2.imwrite(outpath, self.undistort(img))
            else:
                ret, dst=self.corners_unwarp(img)
                if ret == True:
                    outpath=os.path.join(output_folder,"transf_" + os.path.basename(f))
                    cv2.imwrite(outpath, dst)
                    print("tranformed file " + f + " to output " + outpath)
                else:
                    print("failed to transform file " + f)