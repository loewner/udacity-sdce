import pipeline
import numpy as np
import lane
import cv2

class Lanes(object):
    def __init__(self, per, cal, params):
        self.left = lane.Lane(params)
        self.right = lane.Lane(params)
        self.per = per
        self.cal = cal
        self.params = params
        self.binary=None

    def findLanes(self, img):
        # undistort image
        undist = self.cal.undistort(img)
        # uncomment for plotting the selected area for perspective transform
        # copy_undist = np.copy(undist)
        # cv2.polylines(copy_undist, np.int32([points]), 1, (0, 0, 0))

        # gradient and color channel transform to get binary
        self.binary = pipeline.gradientAndColorChannelTransform(undist, self.params)
        # calculate bird's view by perspective transform
        binary_warped = self.per.apply(self.binary)

        # fit polynomial
        if self.left.detected== False or self.right.detected ==False:
            # start from scratch
            print("start fit polynomial from scratch")
            leftx, lefty, rightx, righty = pipeline.find_lane_pixels_from_scratch(binary_warped,self.params)
            self.left.detected = True
            self.right.detected = True
        else:
            # use information from before
            leftx, lefty, rightx, righty = pipeline.find_lane_pixels_posteriori(binary_warped,self.left,self.right,self.params)
            if (len(leftx) == 0) or (len(lefty) == 0) or (len(rightx) == 0) or (len(righty)== 0):
                print("lane lost -> start fit polynomial from scratch")
                leftx, lefty, rightx, righty = pipeline.find_lane_pixels_from_scratch(binary_warped, self.params)


        # Fit a second order polynomial to each using `np.polyfit`
        self.left.current_fit = np.polyfit(lefty, leftx, 2)
        self.right.current_fit = np.polyfit(righty, rightx, 2)

        # add current fit to coeff memory
        self.left.addToCoefficientMemory()
        self.right.addToCoefficientMemory()


        # calc best fit
        self.left.calcBestFit()
        self.right.calcBestFit()

        # calc curvature
        self.left.calcCurvature(img)
        self.right.calcCurvature(img)

        # calc distance to Lane
        self.left.calcDistanceToLine(img)

        # do plotting
        out = self.plotLanesToUndistoredImage(undist)

        return out

    def reset(self):
        self.__init__(self.per, self.cal, self.params)


    def plotLanesToUndistoredImage(self, undist):
        ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
        left_fitx = self.left.best_fit[0] * ploty ** 2 + self.left.best_fit[1] * ploty + self.left.best_fit[2]
        right_fitx = self.right.best_fit[0] * ploty ** 2 + self.right.best_fit[1] * ploty + self.right.best_fit[2]

        # Create an image to draw the lines on
        color_warp = np.zeros_like(undist).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.per.Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        cv2.putText(result, ' Current curvature left lane: ' + str(self.left.radius_of_curvature), \
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'Current curvature right lane: ' + str(self.right.radius_of_curvature), \
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'Distance to center: ' + str(self.left.line_base_pos), \
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # return pipeline.plotLaneToWapredImage(self.right, pipeline.plotLaneToWapredImage(self.left,255*binary_warped))
        # return pipeline.plotLaneToWapredImage(self.right, pipeline.plotLaneToWapredImage(self.left, self.per.apply(undist)))
        # return self.per.apply(undist)
        # return 255*binary_warped

        for i in range(0, 3):
            result[0:200, result.shape[1] - 400:result.shape[1], i] = cv2.resize(255 * self.binary, (400, 200))

        result[200:400, result.shape[1] - 200:result.shape[1], :] = cv2.resize(self.right.plotLaneToWapredImage( \
            self.left.plotLaneToWapredImage(self.per.apply(undist), \
                                            color=(0, 0, 255)), color=(255, 0, 0)), (200, 200))

        return  255*self.binary