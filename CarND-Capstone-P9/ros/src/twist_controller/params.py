#!/usr/bin/env python

import rospy

class CarParameter(object):
	def __init__(self):
		self.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
		self.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
		self.brake_deadband = rospy.get_param('~brake_deadband', .1)
		self.decel_limit = rospy.get_param('~decel_limit', -5)
		self.accel_limit = rospy.get_param('~accel_limit', 1.)
		self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
		self.wheel_base = rospy.get_param('~wheel_base', 2.8498)
		self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
		self.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
		self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)