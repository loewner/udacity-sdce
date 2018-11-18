import numpy as np
import collections
import cv2

class Lane(object):
    def __init__(self, par):

        self.par = par
        # number of frames incluede in best fit
        self.n=self.par.n
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # memory of coefficients
        self.coeffMemory = np.zeros(shape=(0,3))
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def calcCurvature(self, img):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = self.par.ym_per_pix  # meters per pixel in y dimension
        xm_per_pix = self.par.xm_per_pix  # meters per pixel in x dimension
        Anew = xm_per_pix / (ym_per_pix ** 2) * self.best_fit[0]
        Bnew = xm_per_pix / ym_per_pix * self.best_fit[1]
        self.radius_of_curvature = np.round(
            np.sqrt((1 + (2 * Anew * ym_per_pix * img.shape[0] + Bnew) ** 2) ** 3) / np.abs(2 * Anew))

    def calcDistanceToLine(self, img):
        A= self.best_fit[0]
        B = self.best_fit[1]
        C = self.best_fit[2]
        self.line_base_pos = np.round(np.abs(3.7/2 + self.par.xm_per_pix*(A * (img.shape[0])**2 + B * img.shape[0] + C - img.shape[1]/2)), 2)



    def addToCoefficientMemory(self):
        if self.coeffMemory.shape[0] >=self.n:
            self.coeffMemory = self.coeffMemory[0:self.n-1,:]

        self.coeffMemory = np.vstack([self.current_fit.reshape(1,3), self.coeffMemory])


    def calcBestFit(self):
        self.best_fit=np.mean(self.coeffMemory,axis=0)

    def plotLaneToWapredImage(self, warped, thickness=3, color=(200, 32, 0)):
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        A = self.best_fit[0]
        B = self.best_fit[1]
        C = self.best_fit[2]
        try:
            fitx = A * ploty ** 2 + B * ploty + C
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            fitx = 1 * ploty ** 2 + 1 * ploty

        out = np.copy(warped)  # np.dstack((warped,warped,warped))

        # Plots the left and right polynomials on the lane lines
        points_left = np.dstack((fitx, ploty))[0]
        cv2.polylines(out, np.int32([points_left]), 0, color, thickness=thickness)

        return out