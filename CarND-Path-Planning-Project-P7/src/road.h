
#include <math.h>
#include <iostream>
#include <vector>
#include "json.hpp"
#include "spline.h"


using namespace std;

#ifndef MAP_H_
#define MAP_H_

struct waypoint {
		double x;
		double y;
	};

class Road{

public:

	vector<double> map_waypoints_x;
	vector<double> map_waypoints_y;
	vector<double> map_waypoints_s;
	vector<double> map_waypoints_dx;
	vector<double> map_waypoints_dy;

	double max_s;

	tk::spline sx;
	tk::spline sy;
	tk::spline sdx;
	tk::spline sdy;

	vector<int> lanes = {0,1,2};
	double MAX_SPEED = 50.0;

	// For converting back and forth between radians and degrees.
	constexpr double pi() { return M_PI; }
	double deg2rad(double x) { return x * pi() / 180; }
	double rad2deg(double x) { return x * 180 / pi(); }

	/**
	* Constructor
	*/
	Road();
	Road(vector<double> map_waypoints_x, vector<double> map_waypoints_y, vector<double> map_waypoints_s,
			vector<double> map_waypoints_dx, vector<double> map_waypoints_dy, double max_s);

	/**
	* Destructor
	*/
	virtual ~Road();

	double distance(double x1, double y1, double x2, double y2);
	vector<waypoint> getXY_spline(vector<double> s, vector<double> d);

};



#endif /* MAP_H_ */
