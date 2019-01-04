

#ifndef VEHICLE_H_
#define VEHICLE_H_

#include <iostream>
#include <random>
#include <vector>
#include <map>
#include <string>
#include "road.h"
#include <math.h>

using namespace std;

class Vehicle {
public:
	  // some identifier
	  int id;
	  double x;
	  double y;
	  double yaw;
	  double speed;
	  double s;
	  double d;
	  string state;

	  // previous and current path
	  vector<waypoint> previous_path;
	  vector<waypoint> current_path;
	  // begin and end of previous path
	  double begin_path_s;
	  double begin_path_d;
	  double end_path_s;
	  double end_path_d;

	  // ohter vehicles
	  vector<Vehicle> other_veh;

	  // environment
	  Road road;
	  vector<string> states = {"LC", "K", "RC"};
	  double target_speed=1;
	  string target_state="K";
	  double lange_change_length = 50.0;
	  double lane_speeds[3];
	  bool during_lane_change = false;
	  double lange_change_start_s;
	  string lange_change_state; // "LC" or "RC"
	  int lange_change_orientation; // -1 for "LC" and +1 for "RC"
	  int lane_to_change;


	  // other
	  std::map<tuple<int, int>, vector<double>> parameter_JMP;

	  // For converting back and forth between radians and degrees.
		constexpr double pi() { return M_PI; }
		double deg2rad(double x) { return x * pi() / 180; }
		double rad2deg(double x) { return x * 180 / pi(); }

	  /**
	  * Constructor
	  */
	  Vehicle(Road road);
	  Vehicle(int id, double x, double y, double vx, double vy, double s, double d);
	  //Vehicle(int lane, float s, float v, float a, string state="CS");

	  /**
	  * Destructor
	  */
	  virtual ~Vehicle();

	  /**
	   * returns a trajectory of going straight
	   */
	  void drive2(vector<double> speed, vector<double> d);
	  void set_begin_path_s();
	  void set_lane_speeds();
	  void set_target_speed();

	  void start_piloted_driving();
	  void follow();
	  tuple<vector<double>,vector<int>> closest_dist_in_lane(int lane);

	  vector<string> get_successor_states();
	  void set_target_state(vector<string> states);
	  vector<double> inefficiency_cost();
	  vector<double> safety_cost();

	  double curvature(double s);

	  /**
	   * returns the coefficients of the jerk minimizing polynomial
	   */
	  vector<double> compute_paramteters_JMP(vector <double> start, vector <double> end, double T);
	  double eval_JMP( double x, vector<double> params);

	  /**
	   * retruns the current lane
	   *
	   * 0 - left
	   * 1 - middle
	   * 2 - right
	   */
	  int get_lane();

};

#endif /* VEHICLE_H_ */
