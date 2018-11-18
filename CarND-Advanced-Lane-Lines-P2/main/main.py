import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import calibrate
import perspective
import pipeline
import lanes
import params
import lane
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#import PyQt5

runTestImages = True
runFirstVideo = False
runSecondVideo = False
runThirdVideo = False

# load parameters
params = params.Params() # could be extended to load from external file some day ...

# camera calibration
cal = calibrate.Calibrate("../camera_cal/*", 9,6)
# test whether calibration worked
cal.distortionAndTransformPerspective_testFiles("../test_camera_cal")


# perspective transform
points = params.points
# calc perspective transform
per = perspective.Perspective("../test_images/straight_lines1.jpg", points)

lanes = lanes.Lanes(per, cal, params) # reset lanes

# apply pipeline to test images
if runTestImages == True:
    test_images = glob.glob("../test_images/*")
    for f in test_images:

        lanes.reset() # reset lanes

        img = cv2.imread(f)

        out = lanes.findLanes(img)

        cv2.imwrite(os.path.join("../output_images", os.path.basename(f)), out)
        print("transformed test file " + f)



# apply pipeline to first video
lanes.reset()
if runFirstVideo == True:
    white_output = '../output_test_videos/project_video.mp4'
    clip1 = VideoFileClip("../test_videos/project_video.mp4")
    white_clip = clip1.fl_image(lanes.findLanes) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


# apply pipeline to second video
lanes.reset()
if runSecondVideo == True:
    white_output = '../output_test_videos/challenge_video.mp4'
    clip1 = VideoFileClip("../test_videos/challenge_video.mp4")
    white_clip = clip1.fl_image(lanes.findLanes) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


# apply pipeline to third video
lanes.reset()
if runThirdVideo == True:
    white_output = '../output_test_videos/harder_challenge_video.mp4'
    clip1 = VideoFileClip("../test_videos/harder_challenge_video.mp4")
    white_clip = clip1.fl_image(lanes.findLanes) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


#dst=cal.undistort(test_img)
#plt.imshow(test_img)
#cv2.imwrite("tetet.jpg", dst)
#plt.show()




#print(cal.get_images_list())
#test_img = cv2.imread(cal.get_images_list()[4])
#pic = cal.undistort(test_img)
#
#plt.imshow(pic)
#plt.show()

