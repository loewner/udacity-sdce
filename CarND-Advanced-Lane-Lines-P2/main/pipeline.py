import numpy as np
import cv2
import lane
import calibrate
import perspective
import matplotlib.pyplot as plt


def gradDirAndMag(img, thresh_dir, thresh_mag, sobel_kernel=3):
    #Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    #Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    #Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    magnitude = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    #Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh_dir[0]) & (direction <= thresh_dir[1]) & (magnitude >= thresh_mag[0]) & (magnitude <= thresh_mag[1])] = 1

    return binary_output


def gradientAndColorChannelTransform(img, params):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # yellow and white filter
    yellow_binary = np.zeros_like(s_channel)
    yellow_binary[ (h_channel >= params.yellow_h_thresh[0]) & (h_channel <=params.yellow_h_thresh[1]) & \
                   (l_channel >=params.yellow_l_thresh[0]) & (l_channel <=params.yellow_l_thresh[1]) & \
                   (s_channel >= params.yellow_s_thresh[0]) & (s_channel <= params.yellow_s_thresh[1])] = 1
    white_binary = np.zeros_like(s_channel)
    white_binary[ (l_channel >= 200) & (l_channel <= 255) ] = 1


    # derivatives l_channel
    l_sobel = gradDirAndMag(l_channel, params.l_thresh_dir, params.l_thresh_mag, params.sobel_kernel)
    s_sobel = gradDirAndMag(s_channel, params.s_thresh_dir, params.s_thresh_mag, params.sobel_kernel)

    # Threshold color channel
    #s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= params.s_thresh[0]) & (s_channel <= params.s_thresh[1])] = 1

    # combine both binary images
    combined = np.zeros_like(s_channel)
    combined[( ((s_sobel == 1) | (l_sobel == 1) ) & ((yellow_binary ==1) | (white_binary ==1) ) ) | ((s_sobel == 1) & (l_sobel == 1) )] = 1
    #combined[ (yellow_binary == 1) | (white_binary == 1) ] = 1
    #combined[(((l_sobel == 1) & (yellow_binary == 1)  ))] = 1

    ## Sobel x of l_channel
    #l_sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    #abs_l_sobelx = np.absolute(l_sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    #scaled_l_sobel = np.uint8(255 * abs_l_sobelx / np.max(abs_l_sobelx))

    ## Sobel x of s_channel
    #s_sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    #abs_s_sobelx = np.absolute(s_sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    #scaled_s_sobel = np.uint8(255 * abs_s_sobelx / np.max(abs_s_sobelx))

    ## Threshold x gradient
    #lxbinary = np.zeros_like(scaled_l_sobel)
    #lxbinary[(scaled_l_sobel >= sx_thresh[0]) & (scaled_l_sobel <= sx_thresh[1])] = 1
    #sxbinary = np.zeros_like(scaled_s_sobel)
    #sxbinary[(scaled_s_sobel >= sx_thresh[0]) & (scaled_s_sobel <= sx_thresh[1])] = 1

    return combined

def find_lane_pixels_from_scratch(binary_warped, params):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = params.nwindows
        # Set the width of the windows +/- margin
        margin = params.margin
        # Set minimum number of pixels found to recenter window
        minpix = params.minpix

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            #cv2.rectangle(out_img, (win_xleft_low, win_y_low),
            #              (win_xleft_high, win_y_high), (0, 255, 0), 2)
            #cv2.rectangle(out_img, (win_xright_low, win_y_low),
            #              (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty #, out_img

def find_lane_pixels_posteriori(binary_warped, left, right, params):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = params.margin

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit = left.best_fit
    right_fit = right.best_fit

    # Set the area of search based on activated x-values ###
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

    # Fit new polynomials
    #left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ### Visualization ##
    ## Create an image to draw on and an image to show the selection window
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    #window_img = np.zeros_like(out_img)
    ## Color in left and right line pixels
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#
    ## Generate a polygon to illustrate the search window area
    ## And recast the x and y points into usable format for cv2.fillPoly()
    #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
    #                                                                ploty])))])
    #left_line_pts = np.hstack((left_line_window1, left_line_window2))
    #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
    #                                                                 ploty])))])
    #right_line_pts = np.hstack((right_line_window1, right_line_window2))
#
    ## Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
#
    ## Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #### End visualization steps ##




def plotLanesToUndistoredImage(undist, left_lane, right_lane, per):
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    left_fitx = left_lane.best_fit[0] * ploty ** 2 + left_lane.best_fit[1] * ploty + left_lane.best_fit[2]
    right_fitx = right_lane.best_fit[0] * ploty ** 2 + right_lane.best_fit[1] * ploty + right_lane.best_fit[2]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, per.Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    cv2.putText(result, ' Current curvature left lane: ' + str(left_lane.radius_of_curvature), \
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Current curvature right lane: ' + str(right_lane.radius_of_curvature), \
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return result

