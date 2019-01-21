
from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, params):
        # TODO: Implement

        #self.yaw_cntrl = YawController(self.params.wheel_base, self.params.steer_ratio, 0.1, self.params.max_lat_accel, self.params.max_steer_angle)
        #self.lowpass_filter = LowPassFilter(0.02,0.6)

        self.params = params
        
        self.thorttle_cntrl = PID(2,0.0002,0.2 ,0.0, self.params.accel_limit)
        self.yaw_pid_cntrl = PID(3.5,0.000,30,-8.0,8.0)
        
        self.last_time = rospy.get_time()
        self.vehicle_mass = self.params.vehicle_mass
        self.wheel_radius = self.params.wheel_radius
        self.decel_limit = self.params.decel_limit

    def control(self, linear_velocity, angular_velocity, current_velocity, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
        	self.thorttle_cntrl.reset()
        	return 0.0,0.0,0.0

        #throttle, brake, steering = 0.0, 0.0, 0.0
        velocity_err = linear_velocity - current_velocity

        #self.last_velocity = current_velocity
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time

        throttle = self.thorttle_cntrl.step(velocity_err, sample_time)
        brake = 0

        if linear_velocity == 0. and current_velocity < 0.1:
        	throttle = 0
        	brake = 400

        elif throttle < .1 and velocity_err < 0:
        	throttle = 0
        	decel = max(velocity_err, self.decel_limit)
        	brake = abs(decel)*self.vehicle_mass*self.wheel_radius

        #steering = self.yaw_cntrl.get_steering(linear_velocity, angular_velocity, current_velocity)
        steering = self.yaw_pid_cntrl.step(angular_velocity, sample_time)

        # rospy.logwarn(angular_velocity)

        return throttle, brake, steering
        