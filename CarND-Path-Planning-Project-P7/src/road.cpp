
#include <fstream>
#include <math.h>
#include <iostream>
#include <vector>
#include "json.hpp"

#include "road.h"
#include "spline.h"

using namespace std;

/**
* Constructor
*/
Road::Road(){}

Road::~Road(){}

Road::Road(vector<double> map_waypoints_x, vector<double> map_waypoints_y, vector<double> map_waypoints_s,
		vector<double> map_waypoints_dx, vector<double> map_waypoints_dy, double max_s){
	this->map_waypoints_x = map_waypoints_x;
	this->map_waypoints_y = map_waypoints_y;
	this->map_waypoints_s = map_waypoints_s;
	this->map_waypoints_dx = map_waypoints_dx;
	this->map_waypoints_dy = map_waypoints_dy;
	this->max_s = max_s;

	vector<double> extended_waypoints_x;
	vector<double> extended_waypoints_y;
	vector<double> extended_waypoints_s;
	vector<double> extended_waypoints_dx;
	vector<double> extended_waypoints_dy;

	int N = map_waypoints_x.size();
	// append waypoints to -N and 6945.554 + N;
	for (int i = N-5; i<N; i++){
		extended_waypoints_x.push_back(map_waypoints_x[i]);
		extended_waypoints_y.push_back(map_waypoints_y[i]);
		extended_waypoints_s.push_back(map_waypoints_s[i]-this->max_s);
		extended_waypoints_dx.push_back(map_waypoints_dx[i]);
		extended_waypoints_dy.push_back(map_waypoints_dy[i]);
	}
	for (int i = 0; i<N; i++){
		extended_waypoints_x.push_back(map_waypoints_x[i]);
		extended_waypoints_y.push_back(map_waypoints_y[i]);
		extended_waypoints_s.push_back(map_waypoints_s[i]);
		extended_waypoints_dx.push_back(map_waypoints_dx[i]);
		extended_waypoints_dy.push_back(map_waypoints_dy[i]);
	}
	for (int i = 0; i<5; i++){
		extended_waypoints_x.push_back(map_waypoints_x[i]);
		extended_waypoints_y.push_back(map_waypoints_y[i]);
		extended_waypoints_s.push_back(map_waypoints_s[i]+ this->max_s);
		extended_waypoints_dx.push_back(map_waypoints_dx[i]);
		extended_waypoints_dy.push_back(map_waypoints_dy[i]);
	}

	this->sx.set_points(extended_waypoints_s,extended_waypoints_x);
	this->sy.set_points(extended_waypoints_s,extended_waypoints_y);
	this->sdx.set_points(extended_waypoints_s,extended_waypoints_dx);
	this->sdy.set_points(extended_waypoints_s,extended_waypoints_dy);
}


double Road::distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}


vector<waypoint> Road::getXY_spline(vector<double> s, vector<double> d)
{
	vector<waypoint> wypts;

	waypoint current;
	waypoint last;
	last.x = this->sx(2*s[0]-s[1]);
	last.y = this->sy(2*s[0]-s[1]);
	for (int i = 0; i< s.size(); i++){
		current.x = this->sx(s[i]);
		current.y = this->sy(s[i]);

		waypoint normal;
		normal.y = -(current.x - last.x);
		normal.x = current.y - last.y;
		double len = sqrt(normal.y*normal.y + normal.x* normal.x);


		waypoint summed;
		summed.x = current.x + normal.x / len * d[i];
		summed.y = current.y + normal.y / len * d[i];

		wypts.push_back(summed);
		last = current;
	}

	return wypts;

}

