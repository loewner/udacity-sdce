#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
import math
import numpy as np
import uuid
import os

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        # set to true to generate training data 
        self.generate_training_date = False

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        sub7 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.ground_truth_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.colorMap={1: "yellow", 0: "red", 2: "green"}
        
        rospy.spin()


    def ground_truth_cb(self,msg):
    	self.ground_truth = msg.lights

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
        	self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        	self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        
        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
        	self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def distance(self, x1,y1,x2,y2):
    	return math.sqrt((x1-x2)**2+(y1-y2)**2)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        x = pose[0]
        y = pose[1]
        if self.waypoint_tree:
        	return self.waypoint_tree.query([x,y], 1)[1]
        else: 
        	return -1

    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #light = -1

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        #if(self.pose):
        #    car_position = self.get_closest_waypoint([self.pose.pose.position.x, self.pose.pose.position.y])

        #TODO find the closest visible traffic light (if one exists)

        car_x = self.pose.pose.position.x
        car_y = self.pose.pose.position.y

        # len is only 8, so no need to use KD-Trees here
        distances = map(lambda line: self.distance(line[0], line[1], car_x,car_y), stop_line_positions)
        min_index = distances.index(min(distances))

        pose_car = np.array([car_x, car_y])
        closest_light = np.array([stop_line_positions[min_index][0], stop_line_positions[min_index][1]])
        closest_light_prev = np.array([stop_line_positions[min_index-1][0], stop_line_positions[min_index-1][1]])

        if np.dot(closest_light-pose_car, closest_light-closest_light_prev) < 0:
        	min_index = (min_index + 1) % len(distances)

        #rospy.logwarn(self.ground_truth[min_index].state)
        distance_next_light = self.distance(car_x, car_y, stop_line_positions[min_index][0], stop_line_positions[min_index][1])
        #rospy.logwarn("GROUND_TRUTH: next light: " + str(min_index) + " --> " +  self.colorMap[self.ground_truth[min_index].state] + " distance: " + str(distance_next_light) )
        if distance_next_light < 200:
        	light_wp = self.get_closest_waypoint(stop_line_positions[min_index])
        	state = self.get_light_state()
        	#state = self.ground_truth[min_index].state
        	if self.generate_training_date:
        		self.gen_trainig_data(state)
        	return light_wp, state

        return -1, TrafficLight.UNKNOWN

    def gen_trainig_data(self, state):
        # write camera image to file
        folder = "training_data"
        if not os.path.isdir(folder):        	
        	os.mkdir(folder)
        if not os.path.isdir(os.path.join(folder, "IMG")):
        	os.mkdir(os.path.join(folder, "IMG"))
        
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        fn = str(uuid.uuid1()) + ".png"
        fn = os.path.join(folder, "IMG" , fn)
        cv2.imwrite( fn  , cv_image)
        f = open(os.path.join(folder, "labels.txt"),"a") 
        f.write(  fn + "," + str(state) + "\n")
        f.close()



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
